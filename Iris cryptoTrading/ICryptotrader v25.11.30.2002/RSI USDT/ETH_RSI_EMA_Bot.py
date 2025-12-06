import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Configuración
exchange = ccxt.binance({
    'apiKey': 'CHUrW6Pks0Ji5sazJW0Q6sFo2LmipW5qBCzn8JyESoIXGvtMraPyhEEjREMg5sYm',
    'secret': 'uRuwevZGWcQue78V0Tkxgi7PYmFOG9nMPkIxUmHscU2yzVrk0GvtNyHgcSIU4GVZ',
    'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
})

# Parámetros
SYMBOL = 'ETH/USDC'
TIMEFRAME = '5m'
AMOUNT = 10  # 10 USDT
LEVERAGE = 10  # 10x
STOP_LOSS_PCT = 0.02  # 2% SL
TAKE_PROFIT_PCT = 0.03  # 3% TP

def initialize():
    try:
        exchange.load_markets()
        exchange.set_leverage(LEVERAGE, SYMBOL)
        exchange.set_margin_mode('isolated', SYMBOL)
        print(f"✅ {SYMBOL} configurado: {LEVERAGE}x leverage")
    except Exception as e:
        print(f"❌ Error: {e}")

def get_klines(limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = pd.to_numeric(df['close'])
        return df
    except Exception as e:
        print(f"❌ Error obteniendo datos: {e}")
        return None

def calculate_indicators(df):
    # EMA rápida y lenta
    df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def signal_generator(df):
    try:
        ema_fast = df['ema_fast']
        ema_slow = df['ema_slow']
        rsi = df['rsi']
        
        # Cruce alcista de EMAs + RSI > 50
        if (ema_fast.iloc[-2] < ema_slow.iloc[-2] and 
            ema_fast.iloc[-1] > ema_slow.iloc[-1] and 
            rsi.iloc[-1] > 50):
            return 'BUY'
        
        # Cruce bajista de EMAs + RSI < 50
        elif (ema_fast.iloc[-2] > ema_slow.iloc[-2] and 
              ema_fast.iloc[-1] < ema_slow.iloc[-1] and 
              rsi.iloc[-1] < 50):
            return 'SELL'
        
        return None
    except Exception as e:
        print(f"❌ Error en señal: {e}")
        return None

def place_order(signal):
    try:
        current_price = exchange.fetch_ticker(SYMBOL)['last']
        amount = (AMOUNT * LEVERAGE) / current_price
        
        if signal == 'BUY':
            print(f"\n🟢 SEÑAL: COMPRA")
            print(f"💰 ${AMOUNT} | {LEVERAGE}x | Exp: ${AMOUNT * LEVERAGE}")
            
            # Orden de entrada
            order = exchange.create_market_buy_order(SYMBOL, amount)
            entry_price = float(order.get('average', current_price))
            print(f"✅ LONG abierto a ${entry_price:.2f}")
            
            # Stop Loss
            sl_price = entry_price * (1 - STOP_LOSS_PCT)
            exchange.create_order(SYMBOL, 'STOP_MARKET', 'sell', amount,
                                params={'stopPrice': sl_price, 'closePosition': True})
            
            # Take Profit
            tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
            exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'sell', amount,
                                params={'stopPrice': tp_price, 'closePosition': True})
            
            print(f"🎯 TP: ${tp_price:.2f} (+{TAKE_PROFIT_PCT*100}%)")
            print(f"🛑 SL: ${sl_price:.2f} (-{STOP_LOSS_PCT*100}%)")
            
        elif signal == 'SELL':
            print(f"\n🔴 SEÑAL: VENTA")
            print(f"💰 ${AMOUNT} | {LEVERAGE}x | Exp: ${AMOUNT * LEVERAGE}")
            
            # Orden de entrada
            order = exchange.create_market_sell_order(SYMBOL, amount)
            entry_price = float(order.get('average', current_price))
            print(f"✅ SHORT abierto a ${entry_price:.2f}")
            
            # Stop Loss
            sl_price = entry_price * (1 + STOP_LOSS_PCT)
            exchange.create_order(SYMBOL, 'STOP_MARKET', 'buy', amount,
                                params={'stopPrice': sl_price, 'closePosition': True})
            
            # Take Profit
            tp_price = entry_price * (1 - TAKE_PROFIT_PCT)
            exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'buy', amount,
                                params={'stopPrice': tp_price, 'closePosition': True})
            
            print(f"🎯 TP: ${tp_price:.2f} (-{TAKE_PROFIT_PCT*100}%)")
            print(f"🛑 SL: ${sl_price:.2f} (+{STOP_LOSS_PCT*100}%)")
            
    except Exception as e:
        print(f"❌ Error ejecutando orden: {e}")

def check_open_positions():
    try:
        positions = exchange.fetch_positions([SYMBOL])
        for pos in positions:
            if float(pos.get('contracts', 0)) > 0:
                return True
        return False
    except:
        return False

def main():
    initialize()
    print(f"\n{'='*60}")
    print(f"🤖 BOT RSI/EMA - {SYMBOL}")
    print(f"{'='*60}")
    print(f"⏱️ Timeframe: {TIMEFRAME} | Análisis: cada 15 min")
    print(f"💰 Capital: ${AMOUNT} | Apalancamiento: {LEVERAGE}x")
    print(f"🎯 TP: {TAKE_PROFIT_PCT*100}% | 🛑 SL: {STOP_LOSS_PCT*100}%")
    print(f"{'='*60}\n")
    
    while True:
        try:
            print(f"\n⏳ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Verificar posiciones abiertas
            if check_open_positions():
                print("🔄 Posición abierta, esperando...")
                time.sleep(60 * 15)
                continue
            
            # Obtener datos
            df = get_klines()
            if df is None:
                time.sleep(60)
                continue
            
            # Calcular indicadores
            df = calculate_indicators(df)
            
            # Generar señal
            signal = signal_generator(df)
            
            if signal:
                print(f"📡 Señal detectada: {signal}")
                print(f"📊 EMA9: {df['ema_fast'].iloc[-1]:.2f} | EMA21: {df['ema_slow'].iloc[-1]:.2f}")
                print(f"📈 RSI: {df['rsi'].iloc[-1]:.1f}")
                place_order(signal)
                time.sleep(60 * 15)  # Esperar 15 minutos después de operar
            else:
                print("🔍 Sin señal, esperando...")
            
            time.sleep(60 * 15)  # Analizar cada 15 minutos
            
        except KeyboardInterrupt:
            print("\n🔴 Bot detenido")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
