import os
import pandas as pd
import numpy as np
from binance.client import Client
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

API_KEY = os.getenv("BINANCE_API_KEY", "CHUrW6Pks0Ji5sazJW0Q6sFo2LmipW5qBCzn8JyESoIXGvtMraPyhEEjREMg5sYm")
API_SECRET = os.getenv("BINANCE_API_SECRET", "uRuwevZGWcQue78V0Tkxgi7PYmFOG9nMPkIxUmHscU2yzVrk0GvtNyHgcSIU4GVZ")

if not API_KEY or not API_SECRET:
    raise ValueError("Faltan BINANCE_API_KEY o BINANCE_API_SECRET.")

client = Client(API_KEY, API_SECRET)

# Parámetros de modelo
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
LIMIT = 1500
SEQUENCE_LENGTH = 50
FUTURE_HORIZON = 5
UP_THRESHOLD = 0.003
DOWN_THRESHOLD = -0.003
TEST_SIZE = 0.2
VAL_SIZE = 0.1
ENSEMBLE_THRESHOLD = 0.6

# Gestión de riesgo
TP_ATR_MULT = 2.0  # Take Profit = 2 * ATR
SL_ATR_MULT = 1.0  # Stop Loss = 1 * ATR
GARCH_VOL_THRESHOLD = 0.02

# =====================================================
# FUNCIONES DE DATOS
# =====================================================

def get_klines_df(symbol, interval=INTERVAL, limit=LIMIT):
    """Descarga datos OHLCV de Binance Spot"""
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(klines, columns=cols)
    
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for c in numeric_cols:
        df[c] = df[c].astype(float)
    
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    
    return df


def add_arima_garch_features(df, window=300):
    """Añade features basados en ARIMA y GARCH"""
    log_price = np.log(df["close"])
    r = log_price.diff().dropna()
    
    arima_ret_hat = pd.Series(index=df.index, dtype=float)
    garch_vol = pd.Series(index=df.index, dtype=float)
    
    if len(r) < window + 10:
        df["arima_ret_hat"] = 0.0
        df["garch_vol"] = df["log_return"].rolling(20).std().fillna(0.0)
        return df
    
    for i in range(window, len(r)):
        train_r = r.iloc[:i]
        
        # ARIMA
        try:
            arima_model = ARIMA(train_r, order=(1, 0, 1)).fit(method_kwargs={"warn_convergence": False})
            fcast = arima_model.forecast(steps=1)
            df_idx = train_r.index[-1] + 1
            if df_idx in arima_ret_hat.index:
                arima_ret_hat.loc[df_idx] = fcast.iloc[0]
        except Exception:
            pass
        
        # GARCH
        try:
            garch_mod = arch_model(train_r * 100, mean="Zero", vol="GARCH", p=1, q=1)
            garch_res = garch_mod.fit(disp="off")
            cond_vol = garch_res.conditional_volatility
            last_idx = train_r.index[-1]
            if last_idx in garch_vol.index:
                garch_vol.loc[last_idx] = cond_vol.iloc[-1] / 100.0
        except Exception:
            pass
    
    arima_ret_hat = arima_ret_hat.fillna(0.0)
    if garch_vol.isna().all():
        garch_vol = df["log_return"].rolling(20).std().fillna(0.0)
    else:
        garch_vol = garch_vol.fillna(method="ffill").fillna(0.0)
    
    df["arima_ret_hat"] = arima_ret_hat.values
    df["garch_vol"] = garch_vol.values
    
    return df


def add_features_and_labels(df):
    """Añade features técnicos y labels"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    # Retornos
    df["return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))
    
    # EMAs
    df["ema_10"] = close.ewm(span=10, adjust=False).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    period = 14
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Bollinger
    window_bb = 20
    rolling_mean = close.rolling(window_bb).mean()
    rolling_std = close.rolling(window_bb).std()
    df["bb_mid"] = rolling_mean
    df["bb_upper"] = rolling_mean + (2 * rolling_std)
    df["bb_lower"] = rolling_mean - (2 * rolling_std)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    
    # ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()
    
    # ARIMA + GARCH
    df = add_arima_garch_features(df)
    
    # Limpiar NaN
    df = df.dropna().reset_index(drop=True)
    
    # Target / label
    df["future_close"] = df["close"].shift(-FUTURE_HORIZON)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]
    
    def create_label(r):
        if r > UP_THRESHOLD:
            return 1
        elif r < DOWN_THRESHOLD:
            return 0
        else:
            return np.nan
    
    df["label"] = df["future_return"].apply(create_label)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    
    return df


def create_sequences(X, y, seq_len):
    """Crea secuencias para LSTM"""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)


# =====================================================
# CALCULADORA DE INVERSIÓN
# =====================================================

def calcular_inversion(symbol, cantidad_usdt):
    """
    Calcula la inversión basada en predicciones ARIMA/LSTM y stop loss/profit
    
    Args:
        symbol: Par de trading (ej: "BTCUSDT", "ETHUSDC")
        cantidad_usdt: Cantidad a invertir en USDT/USDC
    
    Returns:
        dict con resultados del análisis
    """
    print(f"\n{'='*60}")
    print(f"CALCULADORA DE INVERSIÓN - {symbol}")
    print(f"Cantidad a invertir: ${cantidad_usdt:.2f}")
    print(f"{'='*60}\n")
    
    # 1) Descargar datos
    print("📊 Descargando datos del mercado...")
    df_raw = get_klines_df(symbol)
    print(f"✅ {len(df_raw)} velas descargadas")
    
    # 2) Features + Labels
    print("🔧 Calculando indicadores técnicos...")
    df = add_features_and_labels(df_raw)
    
    if len(df) < SEQUENCE_LENGTH + 50:
        print("❌ Muy pocos datos disponibles")
        return None
    
    feature_cols = [
        "close", "volume",
        "return", "log_return",
        "ema_10", "ema_20", "ema_50",
        "rsi",
        "bb_mid", "bb_upper", "bb_lower", "bb_width",
        "atr",
        "arima_ret_hat",
        "garch_vol"
    ]
    
    X = df[feature_cols].values
    y = df["label"].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3) Secuencias para LSTM
    print("🧠 Entrenando modelo LSTM...")
    X_seq, y_seq = create_sequences(X_scaled, y, SEQUENCE_LENGTH)
    N = len(X_seq)
    test_size = int(N * TEST_SIZE)
    val_size = int(N * VAL_SIZE)
    
    X_train = X_seq[: N - test_size - val_size]
    y_train = y_seq[: N - test_size - val_size]
    
    X_val = X_seq[N - test_size - val_size : N - test_size]
    y_val = y_seq[N - test_size - val_size : N - test_size]
    
    # 4) Modelo LSTM
    tf.keras.backend.clear_session()
    model_lstm = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, X_train.shape[2])),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    
    model_lstm.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )
    
    model_lstm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=64,
        verbose=0
    )
    
    lstm_preds_all = model_lstm.predict(X_seq, verbose=0).flatten()
    df_seq = df.iloc[SEQUENCE_LENGTH:].reset_index(drop=True)
    df_seq["p_lstm"] = lstm_preds_all
    
    # 5) LightGBM
    print("📈 Entrenando modelo LightGBM...")
    X_tab = X_scaled[SEQUENCE_LENGTH:]
    y_tab = y[SEQUENCE_LENGTH:]
    
    X_tab_with_lstm = np.hstack([X_tab, df_seq["p_lstm"].values.reshape(-1, 1)])
    
    from sklearn.model_selection import train_test_split
    X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(
        X_tab_with_lstm, y_tab, test_size=TEST_SIZE, shuffle=False
    )
    
    clf = lgb.LGBMClassifier(
        objective='binary',
        learning_rate=0.01,
        num_leaves=31,
        n_estimators=300,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1
    )
    clf.fit(
        X_train_tab, y_train_tab,
        eval_set=[(X_test_tab, y_test_tab)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # 6) Predicción actual
    last_X = X_tab_with_lstm[-1:] 
    p_lgb_last = clf.predict_proba(last_X)[0, 1]
    p_lstm_last = df_seq["p_lstm"].iloc[-1]
    
    # 7) Datos actuales del mercado
    last_row = df_seq.iloc[-1]
    precio_actual = last_row["close"]
    atr_actual = last_row["atr"]
    rsi_actual = last_row["rsi"]
    garch_vol_actual = last_row["garch_vol"]
    arima_pred = last_row["arima_ret_hat"]
    
    # 8) Calcular Stop Loss y Take Profit
    stop_loss_precio = precio_actual - (SL_ATR_MULT * atr_actual)
    take_profit_precio = precio_actual + (TP_ATR_MULT * atr_actual)
    
    stop_loss_pct = ((stop_loss_precio - precio_actual) / precio_actual) * 100
    take_profit_pct = ((take_profit_precio - precio_actual) / precio_actual) * 100
    
    # 9) Calcular cantidades
    cantidad_monedas = cantidad_usdt / precio_actual
    perdida_maxima = cantidad_usdt * (abs(stop_loss_pct) / 100)
    ganancia_esperada = cantidad_usdt * (take_profit_pct / 100)
    
    # 10) Señal de trading
    long_signal = (p_lstm_last > 0.5) and (p_lgb_last > ENSEMBLE_THRESHOLD)
    short_signal = (p_lstm_last < 0.5) and (p_lgb_last < (1 - ENSEMBLE_THRESHOLD))
    
    if long_signal:
        senal = "🟢 COMPRAR (LONG)"
        confianza = (p_lstm_last + p_lgb_last) / 2
    elif short_signal:
        senal = "🔴 VENDER (SHORT)"
        confianza = (1 - p_lstm_last + (1 - p_lgb_last)) / 2
    else:
        senal = "⚪ ESPERAR / NO ENTRAR"
        confianza = 0.5
    
    # 11) Evaluación de riesgo
    riesgo = "BAJO"
    if garch_vol_actual > GARCH_VOL_THRESHOLD:
        riesgo = "ALTO"
    elif garch_vol_actual > GARCH_VOL_THRESHOLD * 0.7:
        riesgo = "MEDIO"
    
    # 12) Ratio Riesgo/Beneficio
    ratio_rb = abs(take_profit_pct / stop_loss_pct) if stop_loss_pct != 0 else 0
    
    # 13) Mostrar resultados
    print(f"\n{'='*60}")
    print(f"📊 ANÁLISIS DE MERCADO")
    print(f"{'='*60}")
    print(f"Precio actual: ${precio_actual:.4f}")
    print(f"ATR (14): ${atr_actual:.4f}")
    print(f"RSI: {rsi_actual:.2f}")
    print(f"Volatilidad GARCH: {garch_vol_actual*100:.3f}%")
    print(f"Predicción ARIMA (retorno): {arima_pred*100:.3f}%")
    
    print(f"\n{'='*60}")
    print(f"🤖 PREDICCIONES DE MODELOS")
    print(f"{'='*60}")
    print(f"Probabilidad LSTM (subida): {p_lstm_last*100:.2f}%")
    print(f"Probabilidad LightGBM (subida): {p_lgb_last*100:.2f}%")
    print(f"Señal: {senal}")
    print(f"Confianza: {confianza*100:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"💰 CÁLCULO DE INVERSIÓN")
    print(f"{'='*60}")
    print(f"Cantidad a invertir: ${cantidad_usdt:.2f}")
    print(f"Cantidad de {symbol.replace('USDT', '').replace('USDC', '')}: {cantidad_monedas:.6f}")
    
    print(f"\n{'='*60}")
    print(f"🎯 STOP LOSS Y TAKE PROFIT")
    print(f"{'='*60}")
    print(f"Stop Loss:")
    print(f"  - Precio: ${stop_loss_precio:.4f}")
    print(f"  - Porcentaje: {stop_loss_pct:.2f}%")
    print(f"  - Pérdida máxima: ${perdida_maxima:.2f}")
    
    print(f"\nTake Profit:")
    print(f"  - Precio: ${take_profit_precio:.4f}")
    print(f"  - Porcentaje: {take_profit_pct:.2f}%")
    print(f"  - Ganancia esperada: ${ganancia_esperada:.2f}")
    
    print(f"\n{'='*60}")
    print(f"⚖️ EVALUACIÓN DE RIESGO")
    print(f"{'='*60}")
    print(f"Nivel de riesgo: {riesgo}")
    print(f"Ratio Riesgo/Beneficio: 1:{ratio_rb:.2f}")
    
    if ratio_rb >= 2:
        print(f"✅ Excelente ratio R/B (≥2:1)")
    elif ratio_rb >= 1.5:
        print(f"✅ Buen ratio R/B (≥1.5:1)")
    elif ratio_rb >= 1:
        print(f"⚠️ Ratio R/B aceptable (≥1:1)")
    else:
        print(f"❌ Ratio R/B desfavorable (<1:1)")
    
    print(f"\n{'='*60}")
    print(f"📋 RECOMENDACIÓN")
    print(f"{'='*60}")
    
    if long_signal and riesgo != "ALTO" and ratio_rb >= 1.5:
        print(f"✅ RECOMENDADO: Entrar en LONG")
        print(f"   - Comprar {cantidad_monedas:.6f} {symbol.replace('USDT', '').replace('USDC', '')}")
        print(f"   - Colocar Stop Loss en ${stop_loss_precio:.4f}")
        print(f"   - Colocar Take Profit en ${take_profit_precio:.4f}")
    elif short_signal and riesgo != "ALTO" and ratio_rb >= 1.5:
        print(f"✅ RECOMENDADO: Entrar en SHORT (si usas futuros)")
        print(f"   - Vender {cantidad_monedas:.6f} {symbol.replace('USDT', '').replace('USDC', '')}")
        print(f"   - Colocar Stop Loss en ${stop_loss_precio:.4f}")
        print(f"   - Colocar Take Profit en ${take_profit_precio:.4f}")
    elif riesgo == "ALTO":
        print(f"⚠️ NO RECOMENDADO: Volatilidad muy alta")
        print(f"   - Esperar a que la volatilidad disminuya")
    elif ratio_rb < 1.5:
        print(f"⚠️ NO RECOMENDADO: Ratio riesgo/beneficio desfavorable")
        print(f"   - Esperar mejores condiciones de mercado")
    else:
        print(f"⚪ NEUTRAL: No hay señal clara")
        print(f"   - Esperar confirmación del mercado")
    
    print(f"\n{'='*60}\n")
    
    # Retornar resultados
    return {
        "symbol": symbol,
        "precio_actual": precio_actual,
        "cantidad_invertir": cantidad_usdt,
        "cantidad_monedas": cantidad_monedas,
        "stop_loss_precio": stop_loss_precio,
        "stop_loss_pct": stop_loss_pct,
        "perdida_maxima": perdida_maxima,
        "take_profit_precio": take_profit_precio,
        "take_profit_pct": take_profit_pct,
        "ganancia_esperada": ganancia_esperada,
        "p_lstm": p_lstm_last,
        "p_lgb": p_lgb_last,
        "senal": senal,
        "confianza": confianza,
        "riesgo": riesgo,
        "ratio_rb": ratio_rb,
        "rsi": rsi_actual,
        "atr": atr_actual,
        "garch_vol": garch_vol_actual,
        "arima_pred": arima_pred
    }


# =====================================================
# MENÚ INTERACTIVO
# =====================================================

def menu_principal():
    """Menú interactivo para la calculadora"""
    
    # Lista de criptomonedas disponibles
    cryptos_base = ["BTC", "ETH", "XRP", "SOL", "ADA", "LTC", "DOGE", "BNB", "MATIC", "AVAX"]
    
    print("\n" + "="*60)
    print("💰 CALCULADORA DE INVERSIÓN CRYPTO")
    print("   Basada en predicciones ARIMA + LSTM")
    print("="*60)
    
    # 1. Seleccionar criptomoneda
    print("\n📊 CRIPTOMONEDAS DISPONIBLES:")
    for i, crypto in enumerate(cryptos_base, 1):
        print(f"  {i}. {crypto}")
    print(f"  0. Ingresar manualmente")
    
    while True:
        try:
            opcion = input("\nSelecciona una criptomoneda (número): ").strip()
            if opcion == "0":
                crypto = input("Ingresa el símbolo de la criptomoneda (ej: BTC, ETH): ").strip().upper()
                break
            else:
                idx = int(opcion) - 1
                if 0 <= idx < len(cryptos_base):
                    crypto = cryptos_base[idx]
                    break
                else:
                    print("❌ Opción inválida. Intenta de nuevo.")
        except ValueError:
            print("❌ Por favor ingresa un número válido.")
    
    # 2. Seleccionar moneda estable (USDT o USDC)
    print("\n💵 MONEDA ESTABLE:")
    print("  1. USDT")
    print("  2. USDC")
    
    while True:
        try:
            opcion_moneda = input("\nSelecciona la moneda estable (1 o 2): ").strip()
            if opcion_moneda == "1":
                moneda = "USDT"
                break
            elif opcion_moneda == "2":
                moneda = "USDC"
                break
            else:
                print("❌ Opción inválida. Selecciona 1 o 2.")
        except ValueError:
            print("❌ Por favor ingresa 1 o 2.")
    
    # Construir el símbolo
    symbol = f"{crypto}{moneda}"
    
    # 3. Ingresar cantidad a invertir
    while True:
        try:
            cantidad = input(f"\n💰 ¿Cuánto deseas invertir en {moneda}? $").strip()
            cantidad = float(cantidad)
            if cantidad > 0:
                break
            else:
                print("❌ La cantidad debe ser mayor a 0.")
        except ValueError:
            print("❌ Por favor ingresa un número válido.")
    
    # 4. Ejecutar cálculo
    try:
        resultado = calcular_inversion(symbol, cantidad)
        
        # 5. Preguntar si desea hacer otro cálculo
        print("\n¿Deseas calcular otra inversión?")
        print("  1. Sí")
        print("  2. No")
        
        opcion_continuar = input("\nSelecciona una opción (1 o 2): ").strip()
        if opcion_continuar == "1":
            menu_principal()
        else:
            print("\n👋 ¡Gracias por usar la calculadora! ¡Buena suerte con tus inversiones!")
            
    except Exception as e:
        print(f"\n❌ Error al calcular la inversión: {e}")
        print("\nPosibles causas:")
        print("  - El par de trading no existe en Binance")
        print("  - Problemas de conexión con la API")
        print("  - Datos insuficientes para el análisis")
        
        opcion_reintentar = input("\n¿Deseas intentar con otra criptomoneda? (s/n): ").strip().lower()
        if opcion_reintentar == "s":
            menu_principal()
        else:
            print("\n👋 ¡Hasta luego!")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    try:
        menu_principal()
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrumpido por el usuario. ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
