import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import ta  # Technical Analysis library

# Configuración
exchange = ccxt.binance({
    'apiKey': 'CHUrW6Pks0Ji5sazJW0Q6sFo2LmipW5qBCzn8JyESoIXGvtMraPyhEEjREMg5sYm',
    'secret': 'uRuwevZGWcQue78V0Tkxgi7PYmFOG9nMPkIxUmHscU2yzVrk0GvtNyHgcSIU4GVZ',
    'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
})

# Parámetros
SYMBOL = 'LTC/USDT'
TIMEFRAME = '5m'
AMOUNT = 10  # 20 USDT
LEVERAGE = 20  # 20x
MIN_SCORE = 5  # Puntuación mínima para operar (de 10 posibles)

def initialize():
    try:
        exchange.load_markets()
        exchange.set_leverage(LEVERAGE, SYMBOL)
        exchange.set_margin_mode('isolated', SYMBOL)
        print(f"✅ {SYMBOL} configurado: {LEVERAGE}x leverage")
    except Exception as e:
        print(f"❌ Error: {e}")

def get_klines(limit=200):
    """Obtiene más datos para indicadores de largo plazo"""
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['volume'] = pd.to_numeric(df['volume'])
        return df
    except Exception as e:
        print(f"❌ Error obteniendo datos: {e}")
        return None

def calculate_advanced_indicators(df):
    """Calcula TODOS los indicadores técnicos"""
    
    # 1. EMAs (Tendencia)
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # 2. RSI (Momentum)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # 3. MACD (Momentum)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # 4. ADX (Fuerza de Tendencia)
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx.adx()
    
    # 5. ATR (Volatilidad)
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    
    # 6. Bollinger Bands (Volatilidad)
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    
    # 7. Volumen
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # 8. OBV (On-Balance Volume)
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    
    return df

def advanced_signal_generator(df):
    """Sistema de puntuación multi-indicador (0-10 puntos)"""
    try:
        score = 0
        reasons = []
        
        # 1. TENDENCIA PRINCIPAL (2 puntos) - EMA 50 vs 200
        if df['ema_50'].iloc[-1] > df['ema_200'].iloc[-1]:
            score += 2
            reasons.append("✅ Tendencia alcista (EMA50>EMA200)")
        else:
            reasons.append("❌ Tendencia bajista (EMA50<EMA200)")
        
        # 2. FUERZA DE TENDENCIA (2 puntos) - ADX
        adx_value = df['adx'].iloc[-1]
        if adx_value > 25:
            score += 2
            reasons.append(f"✅ Tendencia fuerte (ADX={adx_value:.1f})")
        elif adx_value > 20:
            score += 1
            reasons.append(f"⚠️  Tendencia moderada (ADX={adx_value:.1f})")
        else:
            reasons.append(f"❌ Mercado lateral (ADX={adx_value:.1f})")
        
        # 3. CRUCE DE EMAs (1 punto)
        ema_cross = (df['ema_9'].iloc[-2] < df['ema_21'].iloc[-2] and 
                     df['ema_9'].iloc[-1] > df['ema_21'].iloc[-1])
        if ema_cross:
            score += 1
            reasons.append("✅ Cruce alcista EMA9/21")
        
        # 4. RSI EN ZONA FAVORABLE (1 punto)
        rsi_value = df['rsi'].iloc[-1]
        if 40 < rsi_value < 70:
            score += 1
            reasons.append(f"✅ RSI favorable ({rsi_value:.1f})")
        elif rsi_value < 30:
            reasons.append(f"⚠️  RSI sobreventa ({rsi_value:.1f})")
        else:
            reasons.append(f"❌ RSI sobrecompra ({rsi_value:.1f})")
        
        # 5. MACD ALCISTA (1 punto)
        if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
            score += 1
            reasons.append("✅ MACD alcista")
        
        # 6. VOLUMEN BREAKOUT (2 puntos)
        volume_ratio = df['volume_ratio'].iloc[-1]
        if volume_ratio > 1.5:
            score += 2
            reasons.append(f"✅ Volumen alto ({volume_ratio:.1f}x)")
        elif volume_ratio > 1.2:
            score += 1
            reasons.append(f"⚠️  Volumen moderado ({volume_ratio:.1f}x)")
        else:
            reasons.append(f"❌ Volumen bajo ({volume_ratio:.1f}x)")
        
        # 7. BOLLINGER BANDS (1 punto)
        close_price = df['close'].iloc[-1]
        bb_position = (close_price - df['bb_low'].iloc[-1]) / (df['bb_high'].iloc[-1] - df['bb_low'].iloc[-1])
        if bb_position < 0.3:  # Cerca de banda inferior
            score += 1
            reasons.append("✅ Precio en zona de compra (BB)")
        
        return score, reasons
        
    except Exception as e:
        print(f"❌ Error en señal: {e}")
        return 0, []

def calculate_dynamic_sl_tp(df, entry_price, signal):
    """Calcula SL/TP dinámicos basados en ATR"""
    try:
        atr = df['atr'].iloc[-1]
        
        if signal == 'BUY':
            # Stop Loss: 2 ATRs abajo
            sl_price = entry_price - (atr * 2)
            # Take Profit: 3 ATRs arriba (Risk:Reward 1:1.5)
            tp_price = entry_price + (atr * 3)
        else:  # SELL
            sl_price = entry_price + (atr * 2)
            tp_price = entry_price - (atr * 3)
        
        sl_pct = abs((sl_price - entry_price) / entry_price) * 100
        tp_pct = abs((tp_price - entry_price) / entry_price) * 100
        
        return sl_price, tp_price, sl_pct, tp_pct
        
    except Exception as e:
        print(f"❌ Error calculando SL/TP: {e}")
        # Fallback a valores fijos
        if signal == 'BUY':
            return entry_price * 0.98, entry_price * 1.03, 2.0, 3.0
        else:
            return entry_price * 1.02, entry_price * 0.97, 2.0, 3.0

def place_order(score, reasons, df):
    try:
        current_price = exchange.fetch_ticker(SYMBOL)['last']
        amount = (AMOUNT * LEVERAGE) / current_price
        
        print(f"\n{'='*60}")
        print(f"🎯 SEÑAL DETECTADA - Puntuación: {score}/10")
        print(f"{'='*60}")
        
        # Mostrar razones
        for reason in reasons:
            print(f"  {reason}")
        
        if score < MIN_SCORE:
            print(f"\n⚠️  Puntuación insuficiente ({score}/{MIN_SCORE}). No opero.")
            return
        
        print(f"\n🟢 EJECUTANDO COMPRA")
        print(f"💰 Capital: ${AMOUNT} | Apalancamiento: {LEVERAGE}x")
        print(f"💥 Exposición: ${AMOUNT * LEVERAGE}")
        
        # Orden de entrada
        order = exchange.create_market_buy_order(SYMBOL, amount)
        entry_price = float(order.get('average', current_price))
        print(f"✅ LONG abierto a ${entry_price:.5f}")
        
        # Calcular SL/TP dinámicos
        sl_price, tp_price, sl_pct, tp_pct = calculate_dynamic_sl_tp(df, entry_price, 'BUY')
        
        # Stop Loss
        exchange.create_order(SYMBOL, 'STOP_MARKET', 'sell', amount,
                            params={'stopPrice': sl_price, 'closePosition': True})
        
        # Take Profit
        exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'sell', amount,
                            params={'stopPrice': tp_price, 'closePosition': True})
        
        print(f"\n📊 Gestión de Riesgo (ATR Dinámico):")
        print(f"  🎯 Take Profit: ${tp_price:.5f} (+{tp_pct:.2f}%)")
        print(f"  🛑 Stop Loss: ${sl_price:.5f} (-{sl_pct:.2f}%)")
        print(f"  📈 Risk:Reward = 1:{tp_pct/sl_pct:.2f}")
        print(f"{'='*60}\n")
        
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
    print(f"🤖 BOT RSI/EMA MEJORADO - {SYMBOL}")
    print(f"{'='*60}")
    print(f"⏱️  Timeframe: {TIMEFRAME} | Análisis: cada 15 min")
    print(f"💰 Capital: ${AMOUNT} | Apalancamiento: {LEVERAGE}x")
    print(f"📊 Sistema: Puntuación Multi-Indicador (min {MIN_SCORE}/10)")
    print(f"🎯 SL/TP: Dinámicos basados en ATR")
    print(f"\n📈 Indicadores activos:")
    print(f"  • EMAs: 9, 21, 50, 200")
    print(f"  • RSI, MACD, ADX, ATR")
    print(f"  • Bollinger Bands, OBV")
    print(f"  • Análisis de Volumen")
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
            df = get_klines(limit=200)
            if df is None or len(df) < 200:
                print("⚠️  Datos insuficientes")
                time.sleep(60)
                continue
            
            # Calcular indicadores
            df = calculate_advanced_indicators(df)
            
            # Generar señal con puntuación
            score, reasons = advanced_signal_generator(df)
            
            if score >= MIN_SCORE:
                place_order(score, reasons, df)
                time.sleep(60 * 15)
            else:
                print(f"🔍 Puntuación: {score}/10 - Esperando mejor oportunidad...")
                print(f"  Razones:")
                for reason in reasons[:3]:  # Mostrar top 3
                    print(f"    {reason}")
            
            time.sleep(60 * 15)
            
        except KeyboardInterrupt:
            print("\n🔴 Bot detenido")
            break
        except Exception as e:
            print(f"⚠️  Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
