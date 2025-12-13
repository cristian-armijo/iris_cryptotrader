import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
try:
    from telegram_notifier import send_trade_notification
except ImportError:
    print("⚠️ No se pudo importar telegram_notifier")

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
LEVERAGE = 10
AMOUNT = 6
CONFIDENCE_THRESHOLD = 0.6
STOP_LOSS_PCT = 0.02  # 2%

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

def analyze_forecast(predictions, confidence, current_price):
    try:
        if confidence < CONFIDENCE_THRESHOLD:
            # Cálculo auxiliar para notificación incluso si es hold por confianza baja
            predicted_price = predictions[-1]
            trend = predicted_price - current_price
            price_change_pct = (trend / current_price) * 100
            
            # Valores por defecto para hold
            tp = predicted_price
            if predicted_price > current_price:
                sl = current_price * (1 - STOP_LOSS_PCT)
            else:
                sl = current_price * (1 + STOP_LOSS_PCT)
            
            return "hold", price_change_pct, predicted_price, sl, tp

        predicted_price = predictions[-1]
        trend = predicted_price - current_price
        price_change_pct = (trend / current_price) * 100
        
        print(f"\n📊 Cambio: {price_change_pct:+.2f}% | Confianza: {confidence:.2%}")
        
        if trend > 0 and price_change_pct > 0.05:  # Umbral reducido a 0.05%
            tp = predicted_price
            sl = current_price * (1 - STOP_LOSS_PCT)
            return "buy", price_change_pct, predicted_price, sl, tp
            
        elif trend < 0 and price_change_pct < -0.05:  # Umbral reducido a 0.05%
            tp = predicted_price
            sl = current_price * (1 + STOP_LOSS_PCT)
            return "sell", abs(price_change_pct), predicted_price, sl, tp
            
        else:
            tp = predicted_price
            if trend > 0:
                 sl = current_price * (1 - STOP_LOSS_PCT)
            else:
                 sl = current_price * (1 + STOP_LOSS_PCT)
            return "hold", price_change_pct, predicted_price, sl, tp

    except Exception as e:
        print(f"❌ Error: {e}")
        return "hold", 0, current_price, 0, 0

def execute_trade(signal, change_pct, predictions, current_price, sl, tp):
    try:
        amount = (AMOUNT * LEVERAGE) / current_price
        
        print("\n" + "="*60)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        
        if signal == 'buy':
            print(f"🚀 [SEÑAL DE COMPRA] {SYMBOL}")
            print("="*60)
            print(f"💰 Precio Actual: ${current_price:.4f}")
            print(f"📊 Predicción ARIMA: Tendencia ALCISTA (+{change_pct:.2f}%)")
            print(f"🎯 Precio Objetivo: ${predictions[-1]:.4f}")
            print(f"💵 Inversión: ${AMOUNT} | Apalancamiento: {LEVERAGE}x")
            print(f"💸 Exposición Total: ${AMOUNT * LEVERAGE}")
            
            order = exchange.create_market_buy_order(SYMBOL, amount)
            print(f"✅ Orden de COMPRA ejecutada: {order['id']}")
            
        elif signal == 'sell':
            print(f"🐻 [SEÑAL DE VENTA] {SYMBOL}")
            print("="*60)
            print(f"💰 Precio Actual: ${current_price:.4f}")
            print(f"📊 Predicción ARIMA: Tendencia BAJISTA (-{change_pct:.2f}%)")
            print(f"🎯 Precio Objetivo: ${predictions[-1]:.4f}")
            print(f"💵 Inversión: ${AMOUNT} | Apalancamiento: {LEVERAGE}x")
            print(f"💸 Exposición Total: ${AMOUNT * LEVERAGE}")
            
            order = exchange.create_market_sell_order(SYMBOL, amount)
            print(f"✅ Orden de VENTA ejecutada: {order['id']}")
        
        place_sl_tp_orders(order, signal, sl, tp)
        print("="*60 + "\n")
    except Exception as e:
        print(f"❌ Error: {e}")

def place_sl_tp_orders(order, side, sl_price, tp_price):
    try:
        entry_price = float(order.get('average', 0))
        if entry_price == 0:
             # Fallback si no hay average price
             # Intentar obtener info de la orden de nuevo o usar precio actual
             entry_price = float(exchange.fetch_order(order['id'], SYMBOL)['average'])

        # Calcular porcentajes para mostrar
        tp_change = ((tp_price - entry_price) / entry_price) * 100
        sl_change = ((sl_price - entry_price) / entry_price) * 100
        
        print(f"\n📋 Colocando Stop Loss y Take Profit...")
        
        # Orden Take Profit
        exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'sell' if side == 'buy' else 'buy',
                            order['amount'], params={'stopPrice': tp_price, 'closePosition': True})
        
        # Orden Stop Loss
        exchange.create_order(SYMBOL, 'STOP_MARKET', 'sell' if side == 'buy' else 'buy',
                            order['amount'], params={'stopPrice': sl_price, 'closePosition': True})
        
        print(f"✅ Órdenes colocadas:")
        print(f"   🎯 Take Profit: ${tp_price:.4f} ({tp_change:+.2f}%)")
        print(f"   🛑 Stop Loss: ${sl_price:.4f} ({sl_change:.2f}%)")
        
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

def get_current_price():
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        return ticker['last']
    except Exception as e:
        return None

def main():
    initialize()
    print(f"\n{'='*60}")
    print(f"🤖 BOT ARIMA TELEGRAM - {SYMBOL}")
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
            
            # Obtener precio actual primero
            current_price = get_current_price()
            if current_price is None:
                time.sleep(60)
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
            
            signal, change_pct, predicted_price, sl, tp = analyze_forecast(predictions, confidence, current_price)
            
            # SIEMPRE enviar notificación a Telegram
            try:
                # Calcular cambio % para display si es hold
                if signal == "hold":
                    display_signal = "buy" if predicted_price > current_price else "sell"
                    display_change_pct = abs((predicted_price - current_price) / current_price * 100)
                else:
                    display_signal = signal
                    display_change_pct = change_pct

                send_trade_notification(
                    SYMBOL, display_signal, current_price,
                    (AMOUNT * LEVERAGE) / current_price,
                    sl, tp,
                    f"PRED-{datetime.now().strftime('%H%M%S')}",
                    display_change_pct
                )
                print("📱 Notificación enviada a Telegram")
            except Exception as e:
                print(f"⚠️ Error Telegram: {e}")
            
            if signal != "hold":
                execute_trade(signal, change_pct, predictions, current_price, sl, tp)
            else:
                print("⏸️ Sin señales fuertes (Telegram enviado)")
            
            time.sleep(60 * 10)
        except KeyboardInterrupt:
            print("\n🔴 Detenido")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
