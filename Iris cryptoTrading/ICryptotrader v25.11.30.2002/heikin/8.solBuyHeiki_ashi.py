from binance.client import Client
import pandas as pd
import numpy as np
import time
from datetime import datetime
from ta.momentum import RSIIndicator
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN
# ============================================
api_key = 'CHUrW6Pks0Ji5sazJW0Q6sFo2LmipW5qBCzn8JyESoIXGvtMraPyhEEjREMg5sYm'
api_secret = 'uRuwevZGWcQue78V0Tkxgi7PYmFOG9nMPkIxUmHscU2yzVrk0GvtNyHgcSIU4GVZ'
client = Client(api_key, api_secret)

symbol = 'SOLUSDC'
interval = Client.KLINE_INTERVAL_15MINUTE
quantity = 0.01

# Parámetros de riesgo
STOP_LOSS_PERCENT = 2.5  # 2.5%
TAKE_PROFIT_PERCENT = 6.0  # 6%
ARIMA_ORDER = (5, 1, 0)

# ============================================
# FUNCIONES DE DATOS
# ============================================
def get_data():
    klines = client.get_klines(symbol=symbol, interval=interval, limit=100)
    df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'qav', 'trades', 'tbav', 'tqav', 'ignore'])
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def heikin_ashi(df):
    ha_df = df.copy()
    ha_df['close'] = (df[['open', 'high', 'low', 'close']]).mean(axis=1)
    ha_df['open'] = 0.0
    ha_df.at[0, 'open'] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_df.at[i, 'open'] = (ha_df.at[i-1, 'open'] + ha_df.at[i-1, 'close']) / 2
    ha_df['high'] = df[['high', 'open', 'close']].max(axis=1)
    ha_df['low'] = df[['low', 'open', 'close']].min(axis=1)
    return ha_df

# ============================================
# PREDICCIÓN ARIMA
# ============================================
def predict_with_arima(df, steps=3):
    """Predice precios futuros usando ARIMA"""
    try:
        prices = df['close'].values
        model = ARIMA(prices, order=ARIMA_ORDER)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=steps)
        return forecast
    except Exception as e:
        print(f"⚠️ Error en ARIMA: {e}")
        return None

# ============================================
# MEDIAS MÓVILES
# ============================================
def calculate_emas(df):
    """Calcula EMAs 9, 21, 50"""
    ema9 = df['close'].ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = df['close'].ewm(span=21, adjust=False).mean().iloc[-1]
    ema50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    return ema9, ema21, ema50

def check_ema_trend(ema9, ema21, ema50):
    """Determina tendencia basada en EMAs"""
    if ema9 > ema21 > ema50:
        return "ALCISTA"
    elif ema9 < ema21 < ema50:
        return "BAJISTA"
    else:
        return "LATERAL"

# ============================================
# SEÑALES DE TRADING
# ============================================
def check_ha_signal(df):
    ha = heikin_ashi(df)
    last = ha['close'].iloc[-1] - ha['open'].iloc[-1]
    prev = ha['close'].iloc[-2] - ha['open'].iloc[-2]
    if prev < 0 and last > 0:
        return "BUY"
    elif prev > 0 and last < 0:
        return "SELL"
    return None

def detect_breakout_volume(df):
    recent = df[-20:]
    max_high = recent['high'].max()
    avg_volume = recent['volume'].mean()
    last_volume = df['volume'].iloc[-1]
    current_close = df['close'].iloc[-1]
    if current_close > max_high and last_volume > avg_volume * 1.5:
        return "BUY"
    return None

def check_rsi(df):
    rsi = RSIIndicator(df['close'], window=14).rsi()
    rsi_value = rsi.iloc[-1]
    if rsi_value < 30:
        return "BUY", rsi_value
    elif rsi_value > 70:
        return "SELL", rsi_value
    return None, rsi_value

# ============================================
# CÁLCULO DE STOP LOSS Y TAKE PROFIT
# ============================================
def calculate_sl_tp(current_price, signal, arima_prediction=None):
    """Calcula Stop Loss y Take Profit basado en ARIMA y porcentajes"""
    
    if signal == "BUY":
        # Stop Loss: precio actual - porcentaje
        stop_loss = current_price * (1 - STOP_LOSS_PERCENT / 100)
        
        # Take Profit: usar predicción ARIMA si está disponible, sino porcentaje
        if arima_prediction is not None and len(arima_prediction) > 0:
            predicted_price = arima_prediction[-1]
            # Si la predicción es alcista, usar como referencia
            if predicted_price > current_price:
                take_profit = predicted_price * 1.01  # 1% adicional sobre predicción
            else:
                take_profit = current_price * (1 + TAKE_PROFIT_PERCENT / 100)
        else:
            take_profit = current_price * (1 + TAKE_PROFIT_PERCENT / 100)
            
    elif signal == "SELL":
        # Stop Loss: precio actual + porcentaje
        stop_loss = current_price * (1 + STOP_LOSS_PERCENT / 100)
        
        # Take Profit: usar predicción ARIMA si está disponible, sino porcentaje
        if arima_prediction is not None and len(arima_prediction) > 0:
            predicted_price = arima_prediction[-1]
            # Si la predicción es bajista, usar como referencia
            if predicted_price < current_price:
                take_profit = predicted_price * 0.99  # 1% adicional sobre predicción
            else:
                take_profit = current_price * (1 - TAKE_PROFIT_PERCENT / 100)
        else:
            take_profit = current_price * (1 - TAKE_PROFIT_PERCENT / 100)
    else:
        return None, None
    
    return stop_loss, take_profit

# ============================================
# MENSAJES EN CONSOLA
# ============================================
def print_trading_signal(signal, current_price, stop_loss, take_profit, arima_pred, ema9, ema21, ema50, trend, rsi_value):
    """Imprime mensaje detallado de señal de trading"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*60)
    print(f"[{timestamp}]")
    
    if signal == "BUY":
        print(f"🟢 [SEÑAL DE COMPRA] {symbol}")
    elif signal == "SELL":
        print(f"🔴 [SEÑAL DE VENTA] {symbol}")
    
    print("="*60)
    print(f"💰 Precio Actual: {current_price:.4f} USDC")
    
    if arima_pred is not None and len(arima_pred) > 0:
        pred_price = arima_pred[-1]
        pred_change = ((pred_price - current_price) / current_price) * 100
        print(f"📊 Predicción ARIMA (15min): {pred_price:.4f} USDC ({pred_change:+.2f}%)")
    
    if stop_loss and take_profit:
        sl_change = ((stop_loss - current_price) / current_price) * 100
        tp_change = ((take_profit - current_price) / current_price) * 100
        print(f"🛡️  Stop Loss: {stop_loss:.4f} USDC ({sl_change:.2f}%)") 
        print(f"🎯 Take Profit: {take_profit:.4f} USDC ({tp_change:+.2f}%)")
    
    print(f"📈 EMA 9: {ema9:.2f} | EMA 21: {ema21:.2f} | EMA 50: {ema50:.2f}")
    print(f"📊 RSI: {rsi_value:.2f}")
    print(f"✅ Tendencia: {trend}")
    print(f"🔔 Ejecutando orden de {signal}...")
    print("="*60 + "\n")

def print_status(current_price, ema9, ema21, ema50, trend, rsi_value):
    """Imprime estado actual del mercado"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {symbol} | Precio: {current_price:.4f} | RSI: {rsi_value:.2f} | Tendencia: {trend}")

# ============================================
# ÓRDENES OCO (Stop Loss y Take Profit)
# ============================================
def get_symbol_info():
    """Obtiene información del símbolo para formatear precios correctamente"""
    try:
        info = client.get_symbol_info(symbol)
        # Obtener precisión de precio
        for filter_item in info['filters']:
            if filter_item['filterType'] == 'PRICE_FILTER':
                tick_size = float(filter_item['tickSize'])
                # Calcular decimales necesarios
                decimals = len(str(tick_size).rstrip('0').split('.')[-1])
                return decimals
        return 2  # Default
    except:
        return 2  # Default

def format_price(price, decimals):
    """Formatea el precio con la precisión correcta"""
    return float(f"{price:.{decimals}f}")

def place_oco_order(side, quantity, current_price, stop_loss, take_profit):
    """Coloca orden OCO con stop loss y take profit automáticos"""
    try:
        decimals = get_symbol_info()
        
        # Formatear precios
        tp_price = format_price(take_profit, decimals)
        sl_price = format_price(stop_loss, decimals)
        sl_limit_price = format_price(stop_loss * 0.995, decimals)  # 0.5% por debajo del stop
        
        if side == "BUY":
            # Después de comprar, colocar orden de venta con SL/TP
            print(f"\n📋 Colocando orden OCO de VENTA con SL/TP...")
            oco_order = client.create_oco_order(
                symbol=symbol,
                side='SELL',
                quantity=quantity,
                price=str(tp_price),
                stopPrice=str(sl_price),
                stopLimitPrice=str(sl_limit_price),
                stopLimitTimeInForce='GTC'
            )
            print(f"✅ Orden OCO colocada exitosamente:")
            print(f"   🎯 Take Profit: {tp_price} USDC")
            print(f"   🛡️  Stop Loss: {sl_price} USDC")
            return oco_order
            
        elif side == "SELL":
            # Después de vender, colocar orden de compra con SL/TP
            print(f"\n📋 Colocando orden OCO de COMPRA con SL/TP...")
            oco_order = client.create_oco_order(
                symbol=symbol,
                side='BUY',
                quantity=quantity,
                price=str(tp_price),
                stopPrice=str(sl_price),
                stopLimitPrice=str(sl_limit_price),
                stopLimitTimeInForce='GTC'
            )
            print(f"✅ Orden OCO colocada exitosamente:")
            print(f"   🎯 Take Profit: {tp_price} USDC")
            print(f"   🛡️  Stop Loss: {sl_price} USDC")
            return oco_order
            
    except Exception as e:
        print(f"❌ Error al colocar orden OCO: {e}")
        print(f"⚠️  Continuando sin stop loss/take profit automático...")
        return None

# ============================================
# EJECUCIÓN DE TRADES
# ============================================
def execute_trade(signal, current_price, stop_loss, take_profit, arima_pred, ema9, ema21, ema50, trend, rsi_value):
    """Ejecuta la orden de trading con stop loss y take profit automáticos"""
    
    # Mostrar mensaje detallado
    print_trading_signal(signal, current_price, stop_loss, take_profit, arima_pred, ema9, ema21, ema50, trend, rsi_value)
    
    try:
        if signal == "BUY":
            # Ejecutar compra
            order = client.order_market_buy(symbol=symbol, quantity=quantity)
            print(f"✅ Orden de COMPRA ejecutada: {order['orderId']}")
            print(f"   Cantidad: {quantity} {symbol}")
            
            # Colocar stop loss y take profit automáticamente
            time.sleep(1)  # Esperar un segundo para que se complete la orden
            place_oco_order("BUY", quantity, current_price, stop_loss, take_profit)
            
        elif signal == "SELL":
            # Ejecutar venta
            order = client.order_market_sell(symbol=symbol, quantity=quantity)
            print(f"✅ Orden de VENTA ejecutada: {order['orderId']}")
            print(f"   Cantidad: {quantity} {symbol}")
            
            # Colocar stop loss y take profit automáticamente
            time.sleep(1)  # Esperar un segundo para que se complete la orden
            place_oco_order("SELL", quantity, current_price, stop_loss, take_profit)
            
    except Exception as e:
        print(f"❌ Error al ejecutar orden: {e}")

# ============================================
# LOOP PRINCIPAL
# ============================================
print(f"\n🚀 Bot de Trading Heikin-Ashi + ARIMA iniciado para {symbol}")
print(f"⚙️  Stop Loss: {STOP_LOSS_PERCENT}% | Take Profit: {TAKE_PROFIT_PERCENT}%")
print(f"🔄 Órdenes OCO automáticas: ACTIVADAS")
print(f"⏰ Intervalo: 15 minutos\n")

while True:
    try:
        # Obtener datos
        df = get_data()
        current_price = df['close'].iloc[-1]
        
        # Calcular indicadores
        ema9, ema21, ema50 = calculate_emas(df)
        trend = check_ema_trend(ema9, ema21, ema50)
        rsi_signal, rsi_value = check_rsi(df)
        ha_signal = check_ha_signal(df)
        breakout_signal = detect_breakout_volume(df)
        
        # Predicción ARIMA
        arima_prediction = predict_with_arima(df, steps=3)
        
        # Determinar señal final
        signals = [rsi_signal, ha_signal, breakout_signal]
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        
        final_signal = None
        if buy_count >= 2:
            final_signal = "BUY"
        elif sell_count >= 2:
            final_signal = "SELL"
        
        # Ejecutar trade si hay señal
        if final_signal:
            stop_loss, take_profit = calculate_sl_tp(current_price, final_signal, arima_prediction)
            execute_trade(final_signal, current_price, stop_loss, take_profit, arima_prediction, 
                         ema9, ema21, ema50, trend, rsi_value)
        else:
            # Mostrar estado actual
            print_status(current_price, ema9, ema21, ema50, trend, rsi_value)
        
        # Esperar 15 minutos
        time.sleep(900)
        
    except Exception as e:
        print(f"❌ Error en el loop principal: {str(e)}")
        time.sleep(60)