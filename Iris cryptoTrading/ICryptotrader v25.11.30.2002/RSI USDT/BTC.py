import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import warnings
import json
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow/Keras
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow no está instalado. Por favor ejecuta: pip install tensorflow")

# Configuración de la API
exchange = ccxt.binance({
    'apiKey': 'CHUrW6Pks0Ji5sazJW0Q6sFo2LmipW5qBCzn8JyESoIXGvtMraPyhEEjREMg5sYm',
    'secret': 'uRuwevZGWcQue78V0Tkxgi7PYmFOG9nMPkIxUmHscU2yzVrk0GvtNyHgcSIU4GVZ',
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True
    }
})

# Parámetros de la estrategia
SYMBOL = 'XRP/USDC'
TIMEFRAME = '15m'  # Velas de 15 minutos
LOOKBACK = 60  # 60 velas (15 horas) para predecir
FORECAST_HORIZON = 3  # Predecir las próximas 3 velas (45 minutos)
TRAIN_SIZE = 1000  # Número de velas para entrenamiento inicial
LEVERAGE = 5
AMOUNT = 10
CONFIDENCE_THRESHOLD = 0.6

# Sistema de seguimiento
PREDICTION_LOG_FILE = 'prediction_tracking_xrp_lstm.json'
MODEL_FILE = 'xrp_lstm_model.h5'
prediction_history = []

def load_prediction_history():
    """Carga el historial de predicciones"""
    global prediction_history
    try:
        if os.path.exists(PREDICTION_LOG_FILE):
            with open(PREDICTION_LOG_FILE, 'r') as f:
                prediction_history = json.load(f)
                print(f"📂 Cargadas {len(prediction_history)} predicciones anteriores")
    except Exception as e:
        print(f"⚠️ Error cargando historial: {e}")
        prediction_history = []

def save_prediction_history():
    """Guarda el historial de predicciones"""
    try:
        with open(PREDICTION_LOG_FILE, 'w') as f:
            json.dump(prediction_history, f, indent=2)
    except Exception as e:
        print(f"⚠️ Error guardando historial: {e}")

def add_prediction(predictions, confidence, current_price):
    """Registra una nueva predicción"""
    prediction_entry = {
        'timestamp': datetime.now().isoformat(),
        'current_price': float(current_price),
        'predictions': [float(p) for p in predictions],
        'confidence': float(confidence),
        'model': 'LSTM',
        'timeframe': TIMEFRAME,
        'verified': False
    }
    prediction_history.append(prediction_entry)
    save_prediction_history()
    print(f"📝 Predicción LSTM registrada")

def verify_predictions():
    """Verifica predicciones pasadas"""
    global prediction_history
    
    if not prediction_history:
        return
    
    current_time = datetime.now()
    verified_count = 0
    
    for pred in prediction_history:
        if pred['verified']:
            continue
            
        pred_time = datetime.fromisoformat(pred['timestamp'])
        time_needed = timedelta(minutes=15) * FORECAST_HORIZON
        
        if current_time - pred_time >= time_needed:
            try:
                current_price = get_current_price()
                if current_price is None:
                    continue
                
                last_prediction = pred['predictions'][-1]
                error_pct = abs((current_price - last_prediction) / last_prediction) * 100
                mae = abs(current_price - last_prediction)
                
                direction_predicted = "ALZA" if last_prediction > pred['current_price'] else "BAJA"
                direction_actual = "ALZA" if current_price > pred['current_price'] else "BAJA"
                direction_correct = direction_predicted == direction_actual
                
                pred['verified'] = True
                pred['actual_price'] = float(current_price)
                pred['error_pct'] = float(error_pct)
                pred['mae'] = float(mae)
                pred['direction_correct'] = direction_correct
                
                verified_count += 1
                
                status = "✅ CORRECTO" if direction_correct else "❌ INCORRECTO"
                print(f"\n{'='*60}")
                print(f"📊 VERIFICACIÓN LSTM")
                print(f"{'='*60}")
                print(f"Fecha: {pred_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Precio inicial: ${pred['current_price']:.4f}")
                print(f"Predicción: ${last_prediction:.4f} ({direction_predicted})")
                print(f"Precio real: ${current_price:.4f} ({direction_actual})")
                print(f"Error: {error_pct:.2f}%")
                print(f"Dirección: {status}")
                print(f"{'='*60}\n")
                
            except Exception as e:
                print(f"⚠️ Error verificando: {e}")
    
    if verified_count > 0:
        save_prediction_history()
        display_accuracy_stats()

def display_accuracy_stats():
    """Muestra estadísticas de precisión"""
    verified = [p for p in prediction_history if p.get('verified', False)]
    
    if not verified:
        return
    
    total = len(verified)
    correct_direction = sum(1 for p in verified if p.get('direction_correct', False))
    avg_error = np.mean([p['error_pct'] for p in verified])
    avg_mae = np.mean([p['mae'] for p in verified])
    
    accuracy = (correct_direction / total) * 100
    
    print(f"\n{'='*60}")
    print(f"📈 ESTADÍSTICAS LSTM - XRP")
    print(f"{'='*60}")
    print(f"Total verificadas: {total}")
    print(f"Direcciones correctas: {correct_direction}/{total} ({accuracy:.1f}%)")
    print(f"Error promedio: {avg_error:.2f}%")
    print(f"MAE promedio: ${avg_mae:.4f}")
    print(f"{'='*60}\n")

def initialize():
    """Configuración inicial"""
    try:
        exchange.load_markets()
        exchange.set_leverage(LEVERAGE, SYMBOL)
        exchange.set_margin_mode('isolated', SYMBOL)
        print("✅ Configuración inicial completada")
        load_prediction_history()
    except Exception as e:
        print(f"❌ Error en inicialización: {e}")

def get_historical_data(limit=None):
    """Obtiene datos históricos con OHLCV"""
    try:
        if limit is None:
            limit = TRAIN_SIZE + LOOKBACK + 10
        
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"❌ Error obteniendo datos: {e}")
        return None

def get_current_price():
    """Obtiene precio actual"""
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        return ticker['last']
    except Exception as e:
        print(f"❌ Error obteniendo precio: {e}")
        return None

def prepare_data(df):
    """Prepara datos para LSTM"""
    # Crear features
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=10).std()
    df['volume_change'] = df['volume'].pct_change()
    
    # Rellenar NaN
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    # Seleccionar features
    features = ['close', 'volume', 'returns', 'volatility']
    data = df[features].values
    
    # Normalizar
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler

def create_sequences(data, lookback, forecast_horizon):
    """Crea secuencias para LSTM"""
    X, y = [], []
    
    for i in range(lookback, len(data) - forecast_horizon):
        X.append(data[i-lookback:i])
        # Predecir solo el precio de cierre (columna 0)
        y.append(data[i:i+forecast_horizon, 0])
    
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, output_shape):
    """Construye el modelo LSTM"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(output_shape)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_or_load_model(X_train, y_train):
    """Entrena o carga modelo existente"""
    if os.path.exists(MODEL_FILE):
        print("📂 Cargando modelo existente...")
        try:
            model = load_model(MODEL_FILE)
            print("✅ Modelo cargado")
            return model
        except:
            print("⚠️ Error cargando modelo, entrenando nuevo...")
    
    print("🔨 Entrenando nuevo modelo LSTM...")
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )
    
    model.save(MODEL_FILE)
    print("✅ Modelo entrenado y guardado")
    
    return model

def predict_prices(model, recent_data, scaler):
    """Hace predicciones con el modelo"""
    # Preparar input
    input_data = recent_data[-LOOKBACK:].reshape(1, LOOKBACK, recent_data.shape[1])
    
    # Predecir
    predictions_scaled = model.predict(input_data, verbose=0)[0]
    
    # Desnormalizar solo los precios
    predictions = []
    for pred_scaled in predictions_scaled:
        # Crear array con el formato correcto para inverse_transform
        temp = np.zeros((1, recent_data.shape[1]))
        temp[0, 0] = pred_scaled
        pred_price = scaler.inverse_transform(temp)[0, 0]
        predictions.append(pred_price)
    
    return np.array(predictions)

def calculate_confidence(model, X_recent, y_recent):
    """Calcula confianza basada en MAE reciente"""
    try:
        predictions = model.predict(X_recent[-10:], verbose=0)
        mae = mean_absolute_error(y_recent[-10:].flatten(), predictions.flatten())
        
        # Convertir MAE a confianza (0-1)
        # Menor MAE = mayor confianza
        confidence = 1 / (1 + mae * 100)
        return min(max(confidence, 0), 1)
    except:
        return 0.5

def analyze_forecast(predictions, confidence, current_price):
    """Analiza predicciones para generar señales"""
    try:
        if confidence < CONFIDENCE_THRESHOLD:
            return "hold", 0
        
        trend = predictions[-1] - current_price
        price_change_pct = (trend / current_price) * 100
        
        if trend > 0 and price_change_pct > 0.3:
            return "buy", price_change_pct
        elif trend < 0 and price_change_pct < -0.3:
            return "sell", abs(price_change_pct)
        else:
            return "hold", 0
            
    except Exception as e:
        print(f"❌ Error analizando: {e}")
        return "hold", 0

def execute_trade(signal, change_pct):
    """Ejecuta orden"""
    try:
        current_price = exchange.fetch_ticker(SYMBOL)['last']
        amount = AMOUNT / current_price
        
        if signal == 'buy':
            print(f"🚀 COMPRA (previsión +{change_pct:.2f}%)")
            order = exchange.create_market_buy_order(SYMBOL, amount)
            print(f"✅ Ejecutada a ${current_price:.4f}")
            
        elif signal == 'sell':
            print(f"📉 VENTA (previsión -{change_pct:.2f}%)")
            order = exchange.create_market_sell_order(SYMBOL, amount)
            print(f"✅ Ejecutada a ${current_price:.4f}")
            
    except Exception as e:
        print(f"❌ Error ejecutando: {e}")

def check_open_positions():
    """Verifica posiciones abiertas"""
    try:
        positions = exchange.fetch_positions([SYMBOL])
        for pos in positions:
            if float(pos['contracts']) > 0:
                return True
        return False
    except:
        return False

def main():
    if not TENSORFLOW_AVAILABLE:
        print("❌ No se puede ejecutar sin TensorFlow")
        print("Instala con: pip install tensorflow")
        return
    
    initialize()
    
    print(f"\n{'='*60}")
    print(f"🧠 LSTM Neural Network - {SYMBOL}")
    print(f"{'='*60}")
    print(f"⏱️  Timeframe: {TIMEFRAME}")
    print(f"🔮 Predicción: {FORECAST_HORIZON} velas (45 min)")
    print(f"📊 Lookback: {LOOKBACK} velas")
    print(f"{'='*60}\n")
    
    model = None
    scaler = None
    
    while True:
        try:
            print(f"\n⏳ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            verify_predictions()
            
            if check_open_positions():
                print("🔄 Posición abierta, esperando...")
                time.sleep(60 * 7)
                continue
            
            # Obtener datos
            df = get_historical_data()
            if df is None or len(df) < TRAIN_SIZE:
                print("⚠️ Datos insuficientes")
                time.sleep(60)
                continue
            
            current_price = get_current_price()
            if current_price is None:
                time.sleep(60)
                continue
            
            # Preparar datos
            scaled_data, scaler = prepare_data(df)
            X, y = create_sequences(scaled_data, LOOKBACK, FORECAST_HORIZON)
            
            if len(X) == 0:
                print("⚠️ No hay suficientes datos para secuencias")
                time.sleep(60)
                continue
            
            # Entrenar o cargar modelo
            if model is None:
                model = train_or_load_model(X, y)
            
            # Hacer predicción
            predictions = predict_prices(model, scaled_data, scaler)
            confidence = calculate_confidence(model, X, y)
            
            print(f"🧠 Modelo LSTM | Confianza: {confidence:.2%}")
            print(f"💰 Precio actual: ${current_price:.4f}")
            print(f"🔮 Predicciones: {[f'${p:.4f}' for p in predictions]}")
            
            # Registrar predicción
            add_prediction(predictions, confidence, current_price)
            
            # Analizar y ejecutar
            signal, change_pct = analyze_forecast(predictions, confidence, current_price)
            
            if signal != "hold":
                execute_trade(signal, change_pct)
            else:
                print("🔍 Sin señales suficientes")
            
            time.sleep(60 * 15)
            
        except KeyboardInterrupt:
            print("\n🔴 Detenido")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
