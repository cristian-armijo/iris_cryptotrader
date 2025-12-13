import requests
from datetime import datetime

# Configuración de Telegram
TELEGRAM_BOT_TOKEN = "8507635476:AAEmQeQKSe1v1_FM8J1zTs6eP33dIrHnPcI"
TELEGRAM_CHAT_ID = "7892188422"

def send_telegram_message(message):
    """Envía un mensaje a Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200:
            print("[OK] Notificacion enviada a Telegram")
            return True
        else:
            print(f"[ERROR] Error enviando a Telegram: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error en notificacion Telegram: {e}")
        return False

def send_trade_notification(symbol, signal, current_price, amount, sl, tp, order_id, change_pct):
    """Envía notificación de nueva posición abierta"""
    
    # Determinar dirección
    if signal == 'buy':
        direction = "COMPRA"
        sl_text = f"Stop Loss: ${sl:.4f} (-2.00%)"
        tp_text = f"Take Profit: ${tp:.4f} (+{change_pct:.2f}%)"
    else:
        direction = "VENTA"
        sl_text = f"Stop Loss: ${sl:.4f} (+2.00%)"
        tp_text = f"Take Profit: ${tp:.4f} (-{change_pct:.2f}%)"
    
    # Construir mensaje
    message = f"""
<b>NUEVA POSICION ABIERTA</b>

<b>Simbolo:</b> {symbol}
<b>Direccion:</b> {direction}
<b>Precio:</b> ${current_price:.4f}
<b>Cantidad:</b> {amount:.4f}

<b>CONFIGURA MANUALMENTE:</b>
{sl_text}
{tp_text}

<b>Orden ID:</b> {order_id}
<b>Hora:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return send_telegram_message(message.strip())

def send_test_message():
    """Envía mensaje de prueba"""
    message = "[OK] Bot de Telegram configurado correctamente!"
    return send_telegram_message(message)

if __name__ == "__main__":
    # Probar conexión
    print("Probando conexion con Telegram...")
    send_test_message()
