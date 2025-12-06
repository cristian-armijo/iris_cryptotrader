import ccxt
#push
#push
#push  
import pandas as pd
import numpy as np
import time
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Configuración de la API
exchange = ccxt.binance({
    'apiKey': 'CHUrW6Pks0Ji5sazJW0Q6sFo2LmipW5qBCzn8JyESoIXGvtMraPyhEEjREMg5sYm',
    'secret': 'uRuwevZGWcQue78V0Tkxgi7PYmFOG9nMPkIxUmHscU2yzVrk0GvtNyHgcSIU4GVZ',
    'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
})

# Parámetros
SYMBOL = 'ADA/USDC'
TIMEFRAME = '1h'
TRAIN_WINDOW = 168
FORECAST_HORIZON = 3
LEVERAGE = 60
AMOUNT = 6
CONFIDENCE_THRESHOLD = 0.6

def initialize():
    try:
        exchange.load_markets()
        exchange.set_leverage(LEVERAGE, SYMBOL)
        exchange.set_margin_mode('isolated', SYMBOL)
        print("✅ Configuración completada")
    except Exception as e:
        print(f"❌ Error: {e}")

def get_historical_data():
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=TRAIN_WINDOW + FORECAST_HORIZON + 10)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df['close'].values
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def train_arima_model(data):
    try:
        train_size = int(len(data) * 0.8)
        train, test = data[0:train_size], data[train_size:]
        history = [x for x in train]
        best_params = (1, 1, 1)
        best_rmse = float('inf')
        
        for p in range(1, 3):
            for d in range(0, 2):
                for q in range(1, 3):
                    try:
                        temp_predictions = []
                        for t in range(len(test)):
                            model = ARIMA(history, order=(p, d, q))
                            model_fit = model.fit()
                            yhat = model_fit.forecast()[0]
                            temp_predictions.append(yhat)
                            history.append(test[t])
                        current_rmse = sqrt(mean_squared_error(test, temp_predictions))
                        if current_rmse < best_rmse:
                            best_rmse = current_rmse
                            best_params = (p, d, q)
                    except:
                        continue
        
        final_model = ARIMA(data, order=best_params)
        model_fit = final_model.fit()
        forecast = model_fit.forecast(steps=FORECAST_HORIZON)
        residuals = model_fit.resid
        std_residuals = np.std(residuals)
        confidence = 1 - (std_residuals / np.mean(data[-10:]))
        
        return forecast, confidence, best_params
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, 0, (0, 0, 0)

def analyze_forecast(predictions, confidence):
    try:
        if confidence < CONFIDENCE_THRESHOLD:
            return "hold", 0
        trend = predictions[-1] - predictions[0]
        price_change_pct = (trend / predictions[0]) * 100
        print(f"\n📊 Cambio: {price_change_pct:+.2f}% | Confianza: {confidence:.2%}")
        
        if trend > 0 and price_change_pct > 0.05:  # Umbral reducido a 0.05%
            return "buy", price_change_pct
        elif trend < 0 and price_change_pct < -0.05:  # Umbral reducido a 0.05%
            return "sell", abs(price_change_pct)
        else:
            return "hold", 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return "hold", 0

def execute_trade(signal, change_pct, predictions):
    try:
        current_price = exchange.fetch_ticker(SYMBOL)['last']
        amount = (AMOUNT * LEVERAGE) / current_price
        
        print("\n" + "="*60)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        
        if signal == 'buy':
            print(f"� [SEÑAL DE COMPRA] {SYMBOL}")
            print("="*60)
            print(f"💰 Precio Actual: ${current_price:.4f}")
            print(f"📊 Predicción ARIMA: Tendencia ALCISTA (+{change_pct:.2f}%)")
            print(f"� Precio Objetivo: ${predictions[-1]:.4f}")
            print(f"💵 Inversión: ${AMOUNT} | Apalancamiento: {LEVERAGE}x")
            print(f"💸 Exposición Total: ${AMOUNT * LEVERAGE}")
            
            order = exchange.create_market_buy_order(SYMBOL, amount)
            print(f"✅ Orden de COMPRA ejecutada: {order['id']}")
            
        elif signal == 'sell':
            print(f"� [SEÑAL DE VENTA] {SYMBOL}")
            print("="*60)
            print(f"💰 Precio Actual: ${current_price:.4f}")
            print(f"📊 Predicción ARIMA: Tendencia BAJISTA (-{change_pct:.2f}%)")
            print(f"� Precio Objetivo: ${predictions[-1]:.4f}")
            print(f"💵 Inversión: ${AMOUNT} | Apalancamiento: {LEVERAGE}x")
            print(f"💸 Exposición Total: ${AMOUNT * LEVERAGE}")
            
            order = exchange.create_market_sell_order(SYMBOL, amount)
            print(f"✅ Orden de VENTA ejecutada: {order['id']}")
        
        set_take_profit_stop_loss(order, signal, change_pct, current_price, predictions)
        print("="*60 + "\n")
    except Exception as e:
        print(f"❌ Error: {e}")

def set_take_profit_stop_loss(order, side, change_pct, current_price, predictions):
    try:
        entry_price = float(order.get('average', current_price))
        predicted_price = predictions[1]  # Valor del MEDIO de las 3 predicciones
        
        # Parámetros de riesgo
        STOP_LOSS_PERCENT = 2.0  # 2% de stop loss
        
        if side == 'buy':
            # COMPRA: Esperamos que suba
            # Take Profit: EXACTAMENTE la predicción ARIMA
            if predicted_price > current_price:
                take_profit_price = predicted_price  # SIN margen adicional
            else:
                # Fallback a porcentaje fijo
                take_profit_price = entry_price * (1 + (change_pct / 100))
            
            # Stop Loss: porcentaje fijo
            stop_loss_price = entry_price * (1 - STOP_LOSS_PERCENT / 100)
            
        else:  # sell
            # VENTA: Esperamos que baje
            # Take Profit: EXACTAMENTE la predicción ARIMA
            if predicted_price < current_price:
                take_profit_price = predicted_price  # SIN margen adicional
            else:
                # Fallback a porcentaje fijo
                take_profit_price = entry_price * (1 - (change_pct / 100))
            
            # Stop Loss: porcentaje fijo
            stop_loss_price = entry_price * (1 + STOP_LOSS_PERCENT / 100)
        
        # Calcular porcentajes
        tp_change = ((take_profit_price - entry_price) / entry_price) * 100
        sl_change = ((stop_loss_price - entry_price) / entry_price) * 100
        
        print(f"\n📋 Colocando Stop Loss y Take Profit...")
        
        # Orden Take Profit
        exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'sell' if side == 'buy' else 'buy',
                            order['amount'], params={'stopPrice': take_profit_price, 'closePosition': True})
        
        # Orden Stop Loss
        exchange.create_order(SYMBOL, 'STOP_MARKET', 'sell' if side == 'buy' else 'buy',
                            order['amount'], params={'stopPrice': stop_loss_price, 'closePosition': True})
        
        print(f"✅ Órdenes colocadas:")
        print(f"   🎯 Take Profit: ${take_profit_price:.4f} ({tp_change:+.2f}%)")
        print(f"   �️  Stop Loss: ${stop_loss_price:.4f} ({sl_change:.2f}%)")
        print(f"   📊 Basado en ARIMA: ${predicted_price:.4f}")
    except Exception as e:
        print(f"⚠️ Error TP/SL: {e}")

def check_open_positions():
    try:
        positions = exchange.fetch_positions([SYMBOL])
        for pos in positions:
            if float(pos['contracts']) > 0:
                return True
        return False
    except:
        return False

def main():
    initialize()
    print(f"\n{'='*60}")
    print(f"🤖 BOT ARIMA - {SYMBOL}")
    print(f"{'='*60}")
    print(f"⏱️ {TIMEFRAME} | Horizonte: {FORECAST_HORIZON} velas")
    print(f"💰 ${AMOUNT} | {LEVERAGE}x | Umbral: 0.1% | Análisis: 10 min")
    print(f"{'='*60}\n")
    
    while True:
        try:
            print(f"\n⏳ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if check_open_positions():
                print("🔄 Posición abierta")
                time.sleep(60 * 30)
                continue
            
            data = get_historical_data()
            if data is None:
                time.sleep(60)
                continue
            
            predictions, confidence, params = train_arima_model(data)
            if predictions is None:
                time.sleep(60)
                continue
            
            print(f"📊 ARIMA{params} | Confianza: {confidence:.2%}")
            print(f"🔮 Predicciones: {predictions}")
            
            signal, change_pct = analyze_forecast(predictions, confidence)
            
            if signal != "hold":
                execute_trade(signal, change_pct, predictions)
            else:
                print("⏸️ Sin señales")
            
            time.sleep(60 * 10)
        except KeyboardInterrupt:
            print("\n🔴 Detenido")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
