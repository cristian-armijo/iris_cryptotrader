from binance.client import Client
from binance.enums import *
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
import pandas as pd
import numpy as np
import time
import os
from dotenv import load_dotenv

# Cargar claves
load_dotenv()
api_key = os.getenv("CHUrW6Pks0Ji5sazJW0Q6sFo2LmipW5qBCzn8JyESoIXGvtMraPyhEEjREMg5sYm")
api_secret = os.getenv("uRuwevZGWcQue78V0Tkxgi7PYmFOG9nMPkIxUmHscU2yzVrk0GvtNyHgcSIU4GVZ")

client = Client(api_key, api_secret)

symbol = 'ETHUSDT'
quantity_usdt = 5  # Inversión por operación
leverage = 5

def get_klines(symbol, interval='5m', limit=100):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'qav', 'trades', 'tb_base', 'tb_quote', 'ignore'])
    df['close'] = pd.to_numeric(df['close'])
    return df

def signal_generator(df):
    ema_fast = EMAIndicator(df['close'], window=9).ema_indicator()
    ema_slow = EMAIndicator(df['close'], window=21).ema_indicator()
    rsi = RSIIndicator(df['close'], window=14).rsi()

    if ema_fast.iloc[-2] < ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1] and rsi.iloc[-1] > 50:
        return 'BUY'
    elif ema_fast.iloc[-2] > ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1] and rsi.iloc[-1] < 50:
        return 'SELL'
    else:
        return None

def get_price(symbol):
    return float(client.futures_symbol_ticker(symbol=symbol)['price'])

def place_order(signal):
    price = get_price(symbol)
    quantity = round(quantity_usdt * leverage / price, 3)

    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
    except:
        pass

    if signal == 'BUY':
        order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f'🟢 LONG abierta a {price}')
    elif signal == 'SELL':
        order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f'🔴 SHORT abierta a {price}')

def main_loop():
    print("🚀 Iniciando bot en vivo...")
    while True:
        try:
            df = get_klines(symbol)
            signal = signal_generator(df)
            if signal:
                print(f"📡 Señal detectada: {signal}")
                place_order(signal)
                time.sleep(60 * 5)  # Esperar 5 min antes de la próxima operación
            else:
                print("🔍 Sin señal. Esperando...")
                time.sleep(60)
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main_loop()
