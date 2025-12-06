import os
#push
#test
import pandas as pd
import numpy as np
from binance.client import Client
import lightgbm as lgb  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 

from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import time

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

# ❗ RECOMENDADO: usa variables de entorno, no claves en claro.
# En tu sistema (Windows PowerShell):
#   $env:BINANCE_API_KEY="tu_api_key"
#   $env:BINANCE_API_SECRET="tu_secret"
# O en tu sistema (Windows CMD):
#   set BINANCE_API_KEY=tu_api_key
#   set BINANCE_API_SECRET=tu_secret

# Intenta leer de variables de entorno, si no existen usa las claves hardcoded
API_KEY = os.getenv("BINANCE_API_KEY", "CHUrW6Pks0Ji5sazJW0Q6sFo2LmipW5qBCzn8JyESoIXGvtMraPyhEEjREMg5sYm")
API_SECRET = os.getenv("BINANCE_API_SECRET", "uRuwevZGWcQue78V0Tkxgi7PYmFOG9nMPkIxUmHscU2yzVrk0GvtNyHgcSIU4GVZ")

if not API_KEY or not API_SECRET:
    raise ValueError("Faltan BINANCE_API_KEY o BINANCE_API_SECRET.")

client = Client(API_KEY, API_SECRET)

# PARES A ANALIZAR (puedes añadir versiones USDC si quieres)
SYMBOLS = [
    "XRPUSDT",
    "BTCUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "LTCUSDT",
    "DOGEUSDT",
    # Ejemplo si quieres USDC:
    # "BTCUSDC", "ETHUSDC", ...
]

INTERVAL = Client.KLINE_INTERVAL_15MINUTE  # temporalidad 15m
LIMIT = 1500                              # nº velas a descargar

# Parámetros de modelo / etiquetado
SEQUENCE_LENGTH = 50
FUTURE_HORIZON = 5           # velas hacia adelante para evaluar
UP_THRESHOLD = 0.003         # +0.3% -> label 1
DOWN_THRESHOLD = -0.003      # -0.3% -> label 0
TEST_SIZE = 0.2
VAL_SIZE = 0.1
ENSEMBLE_THRESHOLD = 0.6     # umbral prob. LightGBM

# Gestión de riesgo / backtest
TP_ATR_MULT = 2.0            # TP = 2 * ATR
SL_ATR_MULT = 1.0            # SL = 1 * ATR
GARCH_VOL_THRESHOLD = 0.02   # si vol condicional > 2% filtramos señales
MIN_TRADES_REQUIRED = 30     # mínimo de trades para confiar un poco en EV

# Trading en vivo
LIVE_TRADING = False         # ⚠️ pon True sólo cuando estés SEGURO
QUOTE_ORDER_QTY = 10.0       # comprar / vender 10 USDT o 10 USDC por operación

# =====================================================
# FUNCIONES DE DATOS / DESCARGA
# =====================================================

def get_klines_df(symbol, interval=INTERVAL, limit=LIMIT):
    """
    Descarga datos OHLCV de Binance Spot y devuelve un DataFrame.
    """
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(klines, columns=cols)

    # Convertir a tipos numéricos
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for c in numeric_cols:
        df[c] = df[c].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    return df


def download_klines_to_csv(symbols, interval=INTERVAL, limit=LIMIT, prefix="data"):
    """
    Descarga datos de varios símbolos y los guarda en CSV
    para que luego hagas backtests offline.
    """
    for sym in symbols:
        try:
            df = get_klines_df(sym, interval=interval, limit=limit)
            filename = f"{prefix}_{sym}_{interval}.csv".replace(":", "")
            df.to_csv(filename, index=False)
            print(f"✅ Guardado {filename} ({len(df)} velas)")
            time.sleep(0.2)
        except Exception as e:
            print(f"❌ Error descargando {sym}: {e}")


# =====================================================
# FEATURES: ARIMA + GARCH + TÉCNICOS
# =====================================================

def add_arima_garch_features(df, window=300):
    """
    Añade features basados en ARIMA y GARCH con un esquema walk-forward simple.
    Reduce leakage (aunque es más lento).
    """
    log_price = np.log(df["close"])
    r = log_price.diff().dropna()  # retornos

    arima_ret_hat = pd.Series(index=df.index, dtype=float)
    garch_vol = pd.Series(index=df.index, dtype=float)

    if len(r) < window + 10:
        # muy pocos datos para hacer walk-forward
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
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

# =====================================================
# “ANÁLISIS FUNDAMENTAL” LIGERO (BINANCE 24H)
# =====================================================

def get_fundamental_snapshot(symbol):
    """
    Usa el 24hr ticker de Binance como proxy de fundamental ligero:
    - cambio de precio 24h
    - volumen quotado
    """
    try:
        ticker = client.get_ticker(symbol=symbol)
        price_change_pct = float(ticker.get("priceChangePercent", 0.0))
        quote_volume = float(ticker.get("quoteVolume", 0.0))
        return {
            "priceChangePercent": price_change_pct,
            "quoteVolume": quote_volume
        }
    except Exception as e:
        print(f"Error al obtener fundamental de {symbol}: {e}")
        return {
            "priceChangePercent": 0.0,
            "quoteVolume": 0.0
        }


def fundamental_filter(symbol, direction):
    """
    Filtro fundamental:
    - direction = 'long' o 'short'
    - usa cambio 24h y volumen como filtro básico.
    """
    f = get_fundamental_snapshot(symbol)
    chg = f["priceChangePercent"]
    vol = f["quoteVolume"]

    # mínimo volumen
    if vol < 1e5:  # muy poco volumen
        print(f"❌ Filtro fundamental {symbol}: volumen muy bajo ({vol})")
        return False

    if direction == "long":
        # No comprar si el precio se está desplomando fuerte (ej: < -8%)
        if chg < -8:
            print(f"❌ Filtro fundamental {symbol}: cambio 24h muy negativo ({chg}%) para long")
            return False
    else:  # short o venta
        # No vender/agresivo short si el precio viene demasiado extendido a la baja
        if chg < -15:
            print(f"⚠️ Filtro fundamental {symbol}: ya muy caído ({chg}%), cuidado con shorts")
            # puedes retornar False si quieres ser más estricto

    print(f"✅ Fundamental OK {symbol}: cambio 24h={chg}%, vol={vol}")
    return True

# =====================================================
# EJECUCIÓN DE ÓRDENES SPOT
# =====================================================

def execute_spot_order(symbol, side, quote_amount=QUOTE_ORDER_QTY):
    """
    Ejecuta una orden de mercado en SPOT por un monto en USDT/USDC (quoteOrderQty).
    side = "BUY" o "SELL".
    """
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quoteOrderQty=str(quote_amount)
        )
        print(f"💰 Orden SPOT ejecutada: {side} {quote_amount} de {symbol}")
        print(order)
        return order
    except Exception as e:
        print(f"❌ Error ejecutando orden SPOT {side} {symbol}: {e}")
        return None

# (Para futuros, podrías añadir después client.futures_create_order etc.)

# =====================================================
# PIPELINE COMPLETO POR SÍMBOLO
# =====================================================

def run_pipeline_for_symbol(symbol):
    print(f"\n=====================")
    print(f"Procesando símbolo: {symbol}")
    print(f"=====================")

    # 1) Descargar datos
    df_raw = get_klines_df(symbol)
    print(f"Velas descargadas: {len(df_raw)}")

    # 2) Features + Labels
    df = add_features_and_labels(df_raw)

    if len(df) < SEQUENCE_LENGTH + 50:
        print("Muy pocos datos luego de features/labels, saltando símbolo.")
        return

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
    X_seq, y_seq = create_sequences(X_scaled, y, SEQUENCE_LENGTH)
    N = len(X_seq)
    test_size = int(N * TEST_SIZE)
    val_size = int(N * VAL_SIZE)

    X_train = X_seq[: N - test_size - val_size]
    y_train = y_seq[: N - test_size - val_size]

    X_val = X_seq[N - test_size - val_size : N - test_size]
    y_val = y_seq[N - test_size - val_size : N - test_size]

    X_test = X_seq[N - test_size :]
    y_test = y_seq[N - test_size :]

    print("Shapes LSTM - Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

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
        verbose=1
    )

    lstm_preds_all = model_lstm.predict(X_seq, verbose=0).flatten()
    df_seq = df.iloc[SEQUENCE_LENGTH:].reset_index(drop=True)
    df_seq["p_lstm"] = lstm_preds_all

    # 5) LightGBM sobre features tabulares + p_lstm
    X_tab = X_scaled[SEQUENCE_LENGTH:]
    y_tab = y[SEQUENCE_LENGTH:]

    X_tab_with_lstm = np.hstack([X_tab, df_seq["p_lstm"].values.reshape(-1, 1)])

    X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(
        X_tab_with_lstm, y_tab, test_size=TEST_SIZE, shuffle=False
    )

    train_data = lgb.Dataset(X_train_tab, label=y_train_tab)
    valid_data = lgb.Dataset(X_test_tab, label=y_test_tab)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1
    }

    try:
        lgb_model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            num_boost_round=300,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )
    except TypeError:
        print("⚠️ LightGBM sin callbacks, usando LGBMClassifier.")
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
            eval_set=[(X_train_tab, y_train_tab), (X_test_tab, y_test_tab)],
            early_stopping_rounds=50,
            verbose=50
        )
        lgb_model = clf

    if hasattr(lgb_model, "predict_proba"):
        y_proba_lgb = lgb_model.predict_proba(X_test_tab)[:, 1]
    else:
        y_proba_lgb = lgb_model.predict(X_test_tab)
    y_pred_lgb = (y_proba_lgb > 0.5).astype(int)

    print("\nReporte LightGBM (test):")
    print(classification_report(y_test_tab, y_pred_lgb))

    # 6) Backtest con TP/SL dinámicos por ATR y filtro GARCH
    test_start_index = len(X_train_tab)
    df_test_bt = df_seq.iloc[test_start_index:].reset_index(drop=True)
    df_test_bt["p_lgb"] = y_proba_lgb

    df_test_bt["signal"] = np.where(
        (df_test_bt["p_lstm"] > 0.5) & (df_test_bt["p_lgb"] > ENSEMBLE_THRESHOLD),
        1,
        0
    )

    results = []

    for i in range(len(df_test_bt) - 1):
        row = df_test_bt.iloc[i]

        # Filtro volatilidad GARCH
        if row["garch_vol"] > GARCH_VOL_THRESHOLD:
            continue

        if row["signal"] == 1:
            entry_price = row["close"]
            atr_val = row["atr"]
            if pd.isna(atr_val) or atr_val <= 0:
                continue

            tp_level = entry_price + TP_ATR_MULT * atr_val
            sl_level = entry_price - SL_ATR_MULT * atr_val

            exit_price = None

            for j in range(1, FUTURE_HORIZON + 1):
                if i + j >= len(df_test_bt):
                    break
                future_row = df_test_bt.iloc[i + j]
                high_j = future_row["high"]
                low_j = future_row["low"]

                if high_j >= tp_level:
                    exit_price = tp_level
                    break
                if low_j <= sl_level:
                    exit_price = sl_level
                    break

            if exit_price is None:
                last_idx = min(i + FUTURE_HORIZON, len(df_test_bt) - 1)
                exit_price = df_test_bt.iloc[last_idx]["close"]

            trade_return_pct = (exit_price - entry_price) / entry_price * 100.0
            results.append(trade_return_pct)

    if len(results) == 0:
        print(f"\nNo hubo trades en el backtest para {symbol}")
        return

    results = pd.Series(results)
    winners = results[results > 0]
    losers = results[results < 0]

    win_rate = len(winners) / len(results) if len(results) > 0 else 0
    loss_rate = len(losers) / len(results) if len(results) > 0 else 0
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0
    EV = win_rate * avg_win - loss_rate * avg_loss

    print(f"\n===== RESULTADOS BACKTEST {symbol} =====")
    print(f"N trades: {len(results)}")
    print(f"Win-rate: {win_rate*100:.2f}%")
    print(f"Ganancia media: {avg_win:.3f}%")
    print(f"Pérdida media: {avg_loss:.3f}%")
    print(f"Esperanza matemática (por trade): {EV:.3f}%")
    if len(results) < MIN_TRADES_REQUIRED:
        print(f"⚠️ OJO: sólo {len(results)} trades, poco para confiar en el EV.")

    # 7) Señal actual + filtro fundamental + posible trade en vivo
    last_row = df_test_bt.iloc[-1]
    p_lstm_last = last_row["p_lstm"]
    p_lgb_last = last_row["p_lgb"]

    long_signal = (p_lstm_last > 0.5) and (p_lgb_last > ENSEMBLE_THRESHOLD)
    short_signal = (p_lstm_last < 0.5) and (p_lgb_last < (1 - ENSEMBLE_THRESHOLD))

    accion = "NO ENTRAR / ESPERAR"
    direction = None

    if long_signal:
        accion = "COMPRAR (LONG)"
        direction = "long"
    elif short_signal:
        accion = "VENDER / SHORT (SI USAS FUTUROS)"
        direction = "short"

    print(f"\nSeñal actual para {symbol}: {accion}")
    print(f"p_lstm: {p_lstm_last:.3f} | p_lgb: {p_lgb_last:.3f}")

    if direction and LIVE_TRADING:
        # filtro fundamental
        if fundamental_filter(symbol, direction):
            if direction == "long":
                execute_spot_order(symbol, "BUY", QUOTE_ORDER_QTY)
            else:
                execute_spot_order(symbol, "SELL", QUOTE_ORDER_QTY)
        else:
            print("❌ Señal anulada por filtro fundamental.")

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    # 1) Opcional: descargar data para backtesting offline
    # download_klines_to_csv(SYMBOLS, interval=INTERVAL, limit=LIMIT, prefix="data")

    # 2) Ejecutar pipeline completo por símbolo
    for sym in SYMBOLS:
        run_pipeline_for_symbol(sym)
