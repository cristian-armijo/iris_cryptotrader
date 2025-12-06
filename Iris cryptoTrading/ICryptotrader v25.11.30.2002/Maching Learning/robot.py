# ====== ROBOT DE TRADING CON MACHINE LEARNING ======
# Integra ARIMA + LSTM + Indicadores Técnicos
# Ejecución automática con 10 USDC y 10x apalancamiento
# Versión: 2.0 - ML Enhanced

import os, time, math
from binance.client import Client
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    LSTM_AVAILABLE = True
except:
    print("⚠️ TensorFlow no disponible, solo se usará ARIMA")
    LSTM_AVAILABLE = False

# ====== CONFIGURACIÓN ======
API_KEY = os.getenv("BINANCE_API_KEY", "CHUrW6Pks0Ji5sazJW0Q6sFo2LmipW5qBCzn8JyESoIXGvtMraPyhEEjREMg5sYm")
API_SECRET = os.getenv("BINANCE_API_SECRET", "uRuwevZGWcQue78V0Tkxgi7PYmFOG9nMPkIxUmHscU2yzVrk0GvtNyHgcSIU4GVZ")

# Símbolos a operar
SYMBOLS = ["XRPUSDC", "BTCUSDC", "ETHUSDC"]

# Parámetros de trading
CAPITAL_PER_TRADE = 10  # USDT por operación
LEVERAGE =40  # Apalancamiento 10x
SCORE_THRESHOLD = 7  # Mínimo de puntos para ejecutar (más conservador con ML)
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
LIMIT = 500

# Machine Learning
ML_CONFIDENCE_THRESHOLD = 0.65  # Confianza mínima del modelo
FORECAST_HORIZON = 4  # Predecir 4 velas (1 hora con velas de 15min)
USE_ARIMA = True
USE_LSTM = LSTM_AVAILABLE

# Modo de operación
TEST_MODE = False  # False = ejecutar operaciones reales
SLEEP_BETWEEN = 0.5

# Archivos
LOG_FILE = "trading_ml_log.json"
MODEL_DIR = "ml_models"
trading_log = []

client = Client(API_KEY, API_SECRET)

# Crear directorio para modelos
os.makedirs(MODEL_DIR, exist_ok=True)

# ====== COLORES ======
class C:
    G = "\\033[92m"; Y = "\\033[93m"; R = "\\033[91m"; B = "\\033[94m"
    C = "\\033[96m"; M = "\\033[95m"; N = "\\033[0m"; BOLD = "\\033[1m"

# ====== FUNCIONES ML ======
def train_arima(data, order=(2,1,2)):
    """Entrena modelo ARIMA y hace predicciones"""
    try:
        log_data = np.log(data + 1e-10)
        model = ARIMA(log_data, order=order)
        model_fit = model.fit()
        forecast_log = model_fit.forecast(steps=FORECAST_HORIZON)
        forecast = np.exp(forecast_log)
        
        # Confianza basada en residuos
        residuals = model_fit.resid
        confidence = 1 / (1 + np.std(residuals))
        
        return forecast, confidence
    except Exception as e:
        print(f"{C.R}Error ARIMA: {e}{C.N}")
        return None, 0

def create_lstm_model(lookback, features=1):
    """Crea arquitectura LSTM"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(FORECAST_HORIZON)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_lstm_data(data, lookback=60):
    """Prepara datos para LSTM"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data) - FORECAST_HORIZON):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i:i+FORECAST_HORIZON, 0])
    
    return np.array(X), np.array(y), scaler

def train_lstm(data, symbol):
    """Entrena o carga modelo LSTM"""
    try:
        model_path = os.path.join(MODEL_DIR, f"{symbol.replace('/', '_')}_lstm.h5")
        lookback = 60
        
        # Preparar datos
        X, y, scaler = prepare_lstm_data(data, lookback)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Cargar o crear modelo
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = create_lstm_model(lookback)
            model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            model.save(model_path)
        
        # Hacer predicción
        last_sequence = data[-lookback:]
        scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1))
        scaled_sequence = scaled_sequence.reshape((1, lookback, 1))
        
        prediction_scaled = model.predict(scaled_sequence, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
        
        # Confianza basada en varianza de predicciones
        confidence = 1 / (1 + np.std(prediction) / np.mean(prediction))
        
        return prediction, confidence
    except Exception as e:
        print(f"{C.R}Error LSTM: {e}{C.N}")
        return None, 0

# ====== FUNCIONES AUXILIARES ======
def log_trade(symbol, action, score, price, predicted_price, sl, tp, reason):
    """Registra operación"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "action": action,
        "score": score,
        "entry_price": price,
        "predicted_price": predicted_price,
        "sl": sl,
        "tp": tp,
        "reason": reason
    }
    trading_log.append(entry)
    try:
        with open(LOG_FILE, 'w') as f:
            json.dump(trading_log, f, indent=2)
    except Exception as e:
        print(f"{C.R}Error guardando log: {e}{C.N}")

def get_klines(symbol, interval=INTERVAL, limit=LIMIT):
    """Obtiene velas históricas"""
    try:
        kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        cols = ['time','open','high','low','close','volume','ct','qav','trades','tbav','tqav','ignore']
        df = pd.DataFrame(kl, columns=cols)
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df[['time','open','high','low','close','volume']]
    except Exception as e:
        print(f"{C.R}Error obteniendo datos: {e}{C.N}")
        return None

def add_indicators(df):
    """Añade indicadores técnicos"""
    df = df.copy()
    df['ema20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema200'] = EMAIndicator(df['close'], window=200).ema_indicator()
    df['rsi14'] = RSIIndicator(df['close'], window=14).rsi()
    
    macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_sig'] = macd.macd_signal()
    
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr14'] = atr.average_true_range()
    
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_up'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    
    return df

def calculate_ml_score(current_price, predictions, confidence):
    """Calcula puntuación ML (0-3 puntos)"""
    if predictions is None or confidence < ML_CONFIDENCE_THRESHOLD:
        return 0, "Confianza insuficiente"
    
    predicted_price = predictions[-1]
    change_pct = ((predicted_price - current_price) / current_price) * 100
    
    score = 0
    reason = []
    
    # Puntos por magnitud del cambio predicho
    if abs(change_pct) > 2:
        score += 2
        reason.append(f"Cambio predicho: {change_pct:+.2f}%")
    elif abs(change_pct) > 1:
        score += 1
        reason.append(f"Cambio predicho: {change_pct:+.2f}%")
    
    # Punto adicional por alta confianza
    if confidence > 0.8:
        score += 1
        reason.append(f"Alta confianza: {confidence:.2%}")
    
    return score, " | ".join(reason), predicted_price, change_pct

def calculate_technical_score(row):
    """Sistema de puntuación técnica (0-7 puntos)"""
    close = row['close']
    ema20, ema50, ema200 = row['ema20'], row['ema50'], row['ema200']
    rsi = row['rsi14']
    macd, macd_sig = row['macd'], row['macd_sig']
    vol, vol_ma = row['volume'], row['vol_ma20']
    bb_mid = row['bb_mid']
    
    long_score = 0
    short_score = 0
    long_reasons = []
    short_reasons = []
    
    # LONG
    if close > ema20 > ema50:
        long_score += 2
        long_reasons.append("Precio > EMA20 > EMA50")
    if 50 < rsi < 70:
        long_score += 1
        long_reasons.append("RSI alcista")
    if macd > macd_sig:
        long_score += 1
        long_reasons.append("MACD > Signal")
    if vol > vol_ma:
        long_score += 1
        long_reasons.append("Volumen confirmado")
    if close > ema200:
        long_score += 1
        long_reasons.append("Sesgo alcista")
    if close > bb_mid:
        long_score += 1
        long_reasons.append("Precio > BB Media")
    
    # SHORT
    if close < ema20 < ema50:
        short_score += 2
        short_reasons.append("Precio < EMA20 < EMA50")
    if 30 < rsi < 50:
        short_score += 1
        short_reasons.append("RSI bajista")
    if macd < macd_sig:
        short_score += 1
        short_reasons.append("MACD < Signal")
    if vol > vol_ma:
        short_score += 1
        short_reasons.append("Volumen confirmado")
    if close < ema200:
        short_score += 1
        short_reasons.append("Sesgo bajista")
    if close < bb_mid:
        short_score += 1
        short_reasons.append("Precio < BB Media")
    
    return {
        "long_score": long_score,
        "short_score": short_score,
        "long_reasons": long_reasons,
        "short_reasons": short_reasons
    }

def determine_signal(tech_scores, ml_score, ml_change_pct):
    """Determina señal combinando técnico + ML"""
    total_long = tech_scores['long_score'] + (ml_score if ml_change_pct > 0 else 0)
    total_short = tech_scores['short_score'] + (ml_score if ml_change_pct < 0 else 0)
    
    all_reasons = []
    
    if total_long >= SCORE_THRESHOLD and total_long > total_short + 2:
        all_reasons = tech_scores['long_reasons']
        return "BUY", total_long, all_reasons
    elif total_short >= SCORE_THRESHOLD and total_short > total_long + 2:
        all_reasons = tech_scores['short_reasons']
        return "SELL", total_short, all_reasons
    else:
        return "WAIT", max(total_long, total_short), []

def execute_order(symbol, signal, predicted_price, current_price, atr):
    """Ejecuta orden con TP en precio predicho"""
    if TEST_MODE:
        print(f"{C.Y}[MODO TEST] No se ejecuta orden real{C.N}")
        return True, predicted_price, current_price - atr
    
    try:
        # Configurar apalancamiento
        client.futures_change_leverage(symbol=symbol.replace("USDC", "USDT"), leverage=LEVERAGE)
        
        # Calcular cantidad
        quantity = round((CAPITAL_PER_TRADE * LEVERAGE) / current_price, 4)
        
        # Ejecutar orden
        if signal == "BUY":
            order = client.futures_create_order(
                symbol=symbol.replace("USDC", "USDT"),
                side="BUY",
                type="MARKET",
                quantity=quantity
            )
            sl = current_price - atr
            tp = predicted_price
        else:
            order = client.futures_create_order(
                symbol=symbol.replace("USDC", "USDT"),
                side="SELL",
                type="MARKET",
                quantity=quantity
            )
            sl = current_price + atr
            tp = predicted_price
        
        print(f"{C.G}✓ Orden ejecutada{C.N}")
        
        # Configurar TP/SL
        client.futures_create_order(
            symbol=symbol.replace("USDC", "USDT"),
            side="SELL" if signal == "BUY" else "BUY",
            type="TAKE_PROFIT_MARKET",
            stopPrice=round(tp, 4),
            closePosition=True
        )
        
        client.futures_create_order(
            symbol=symbol.replace("USDC", "USDT"),
            side="SELL" if signal == "BUY" else "BUY",
            type="STOP_MARKET",
            stopPrice=round(sl, 4),
            closePosition=True
        )
        
        print(f"{C.G}✓ TP/SL configurados{C.N}")
        return True, tp, sl
        
    except Exception as e:
        print(f"{C.R}✗ Error ejecutando: {e}{C.N}")
        return False, None, None

def analyze_symbol(symbol):
    """Análisis completo con ML + Técnico"""
    print(f"\\n{C.BOLD}{C.C}{'='*70}{C.N}")
    print(f"{C.BOLD}{C.C}📊 ANÁLISIS ML: {symbol} | {datetime.now().strftime('%H:%M:%S')}{C.N}")
    print(f"{C.C}{'='*70}{C.N}")
    
    # Obtener datos
    df = get_klines(symbol)
    if df is None or len(df) < 200:
        print(f"{C.R}Datos insuficientes{C.N}")
        return
    
    # Indicadores
    df = add_indicators(df)
    row = df.iloc[-1]
    close_prices = df['close'].values
    
    # Machine Learning
    ml_score = 0
    ml_reason = ""
    predicted_price = close_prices[-1]
    ml_change_pct = 0
    
    if USE_ARIMA:
        print(f"{C.Y}🔮 Entrenando ARIMA...{C.N}")
        arima_pred, arima_conf = train_arima(close_prices[-200:])
        if arima_pred is not None:
            ml_s, ml_r, pred_p, chg = calculate_ml_score(row['close'], arima_pred, arima_conf)
            ml_score = max(ml_score, ml_s)
            if ml_s > 0:
                ml_reason = ml_r
                predicted_price = pred_p
                ml_change_pct = chg
            print(f"{C.G}✓ ARIMA: Confianza {arima_conf:.2%} | Predicción: ${arima_pred[-1]:.4f}{C.N}")
    
    if USE_LSTM:
        print(f"{C.Y}🧠 Entrenando LSTM...{C.N}")
        lstm_pred, lstm_conf = train_lstm(close_prices, symbol)
        if lstm_pred is not None:
            ml_s, ml_r, pred_p, chg = calculate_ml_score(row['close'], lstm_pred, lstm_conf)
            if ml_s > ml_score:
                ml_score = ml_s
                ml_reason = ml_r
                predicted_price = pred_p
                ml_change_pct = chg
            print(f"{C.G}✓ LSTM: Confianza {lstm_conf:.2%} | Predicción: ${lstm_pred[-1]:.4f}{C.N}")
    
    # Análisis técnico
    tech_scores = calculate_technical_score(row)
    
    # Determinar señal
    signal, total_score, reasons = determine_signal(tech_scores, ml_score, ml_change_pct)
    
    # Mostrar resultados
    print(f"\\n{C.BOLD}💰 Precio Actual:{C.N} ${row['close']:.4f}")
    print(f"{C.BOLD}🎯 Precio Predicho:{C.N} ${predicted_price:.4f} ({ml_change_pct:+.2f}%)")
    print(f"{C.BOLD}📊 Score Técnico:{C.N} LONG {tech_scores['long_score']}/7 | SHORT {tech_scores['short_score']}/7")
    print(f"{C.BOLD}🤖 Score ML:{C.N} {ml_score}/3 - {ml_reason}")
    print(f"{C.BOLD}🎯 Score Total:{C.N} {total_score}/10")
    
    # Señal
    if signal == "BUY":
        color = C.G
        icon = "🟢"
    elif signal == "SELL":
        color = C.R
        icon = "🔴"
    else:
        color = C.Y
        icon = "🟡"
    
    print(f"\\n{C.BOLD}🚦 SEÑAL:{C.N} {color}{icon} {signal}{C.N}")
    
    if reasons:
        print(f"\\n{C.BOLD}📋 Razones:{C.N}")
        for r in reasons:
            print(f"  ✓ {r}")
    
    # Ejecutar si hay señal
    if signal in ["BUY", "SELL"]:
        print(f"\\n{C.BOLD}{C.M}🤖 EJECUTANDO OPERACIÓN...{C.N}")
        print(f"  Capital: ${CAPITAL_PER_TRADE}")
        print(f"  Apalancamiento: {LEVERAGE}x")
        print(f"  TP (Precio Predicho): ${predicted_price:.4f}")
        print(f"  SL (ATR): ${row['close'] - row['atr14']:.4f}")
        
        success, tp, sl = execute_order(symbol, signal, predicted_price, row['close'], row['atr14'])
        
        if success:
            log_trade(symbol, signal, total_score, row['close'], predicted_price, sl, tp, " | ".join(reasons))
            print(f"{C.G}✓ Operación registrada{C.N}")

def main():
    """Loop principal"""
    print(f"{C.BOLD}{C.C}")
    print("="*70)
    print("  🤖 ROBOT DE TRADING CON MACHINE LEARNING")
    print("  ARIMA + LSTM + Indicadores Técnicos")
    print("  Versión 2.0")
    print("="*70)
    print(C.N)
    
    print(f"\\n{C.Y}Modo: {'TEST' if TEST_MODE else 'LIVE (OPERACIONES REALES)'}{C.N}")
    print(f"Símbolos: {', '.join(SYMBOLS)}")
    print(f"Capital: ${CAPITAL_PER_TRADE} | Apalancamiento: {LEVERAGE}x")
    print(f"Score mínimo: {SCORE_THRESHOLD}/10")
    print(f"ML: ARIMA={USE_ARIMA} | LSTM={USE_LSTM}")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            print(f"\\n{C.BOLD}{C.B}{'='*70}")
            print(f"  ITERACIÓN #{iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}{C.N}")
            
            for symbol in SYMBOLS:
                try:
                    analyze_symbol(symbol)
                    time.sleep(SLEEP_BETWEEN)
                except Exception as e:
                    print(f"{C.R}Error: {e}{C.N}")
            
            print(f"\\n{C.Y}⏳ Esperando 15 minutos...{C.N}")
            time.sleep(900)  # 15 minutos
            
    except KeyboardInterrupt:
        print(f"\\n{C.Y}🛑 Robot detenido{C.N}")
        print(f"Operaciones: {len(trading_log)}")

if __name__ == "__main__":
    main()
