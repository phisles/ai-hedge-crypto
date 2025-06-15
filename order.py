import os
import json
import requests
from dotenv import load_dotenv
import csv
import sys
sys.path.append("/root/stock2")
from config2 import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORDER_FILE = os.path.join(SCRIPT_DIR, "order-data", "alpaca_crypto_order_output.json")
CSV_LOG = os.path.join(SCRIPT_DIR, "order-data", "alpaca_order_log_crypto.csv")

TESTING_MODE = True  # Set to False in production

# Create CSV with headers if it doesn't exist yet
os.makedirs(os.path.dirname(CSV_LOG), exist_ok=True)

if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "json_side", "qty", "alpaca_side", "limit_price", "order_id", "status", "created_at"])

API_KEY = APCA_API_KEY_ID
API_SECRET = APCA_API_SECRET_KEY
BASE_URL = APCA_API_BASE_URL

headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type": "application/json"
}

def print_current_holdings():
    print("\n📦 CURRENT HOLDINGS SNAPSHOT")
    try:
        r = requests.get(f"{BASE_URL}/v2/positions", headers=headers)

        positions = r.json()
        if not positions:
            print("🔸 No current positions.")
        for p in positions:
            print(f"   🔹 {p['symbol']}: {p['qty']} shares @ avg price ${p['avg_entry_price']}")
    except Exception as e:
        print(f"❌ Error fetching positions: {e}")

def get_current_position(symbol):
    try:
        r = requests.get(f"{BASE_URL}/v2/positions/{symbol}", headers=headers)
        if r.status_code == 404:
            return 0

        return float(r.json().get("qty", 0))
    except Exception as e:
        print(f"⚠️ Couldn't fetch position for {symbol}: {e}")
        return 0

def get_limit_price(symbol, side):
    url = f"https://data.alpaca.markets/v1beta3/crypto/us/latest/quotes?symbols={symbol}"
    print(f"\n🌐 Fetching quote for {symbol}")
    print(f"🔗 URL: {url}")

    try:
        r = requests.get(url, headers=headers)
        print(f"📬 Status Code: {r.status_code}")
        print(f"📦 Response: {r.text}")

        r.raise_for_status()

        data = r.json()
        quote = data.get("quotes", {}).get(symbol)

        if not quote:
            raise ValueError(f"No quote data found for {symbol}")

        ask = quote.get("ap")
        bid = quote.get("bp")

        if side in ("buy", "cover"):
            print(f"✅ Ask: {ask}")
            return round(ask, 4) if ask else None
        else:
            print(f"✅ Bid: {bid}")
            return round(bid, 4) if bid else None

    except Exception as e:
        print(f"❌ Failed to fetch quote for {symbol}: {e}")
        return None

def submit_order(symbol, qty, side):
    alpaca_side = "buy" if side in ("buy", "cover") else "sell"
    limit_price = get_limit_price(symbol, side)

    if limit_price is None:
        print(f"⚠️ Skipping {symbol} - could not fetch limit price.")
        return

    print(f"\n📄 Instruction: {side.upper()} {symbol} x{qty}")
    print(f"🧾 Order JSON Input: {json.dumps({'symbol': symbol, 'qty': qty, 'side': side}, indent=2)}")

    if side == "short":
        current_qty = get_current_position(symbol)
        if current_qty > 0:
            print(f"📊 You currently own {current_qty} shares of {symbol}. Increasing sell qty to short.")
            print(f"📉 Final Qty to Sell = Owned {int(current_qty)} + Short {qty} = {int(current_qty + qty)}")
            qty += int(current_qty)

    # Cancel conflicting sell orders if submitting a buy to prevent wash trade
    if side == "buy":
        try:
            open_orders = requests.get(f"{BASE_URL}/v2/orders?status=open", headers=headers)
            open_orders.raise_for_status()
            for o in open_orders.json():
                if o["symbol"] == symbol and o["side"] == "sell":
                    print(f"⚠️ Detected open SELL order for {symbol}. Canceling to avoid wash trade...")
                    cancel = requests.delete(f"{BASE_URL}/v2/orders/{o['id']}", headers=headers)
                    cancel.raise_for_status()
                    print(f"✅ Canceled conflicting SELL order {o['id']}")
        except Exception as e:
            print(f"❌ Failed to check/cancel conflicting sell order: {e}")

    print(f"🔢 Will submit as Alpaca side='{alpaca_side}' with limit price = {limit_price:.2f}")

    order_payload = {
        "symbol": symbol,
        "qty": qty,
        "side": alpaca_side,
        "type": "limit",
        "limit_price": limit_price,
        "time_in_force": "gtc"
    }

    try:
        r = requests.post(f"{BASE_URL}/v2/orders", headers=headers, json=order_payload)

        try:
            r.raise_for_status()
            order_result = r.json()

            order_id = order_result.get("id")
            status = order_result.get("status")
            created_at = order_result.get("created_at")

            print(f"✅ Order submitted successfully:")
            print(f"   🔹 Order ID: {order_id}")
            print(f"   🔹 Status: {status}")
            print(f"   🔹 Submitted Price: {order_result.get('limit_price')}")
            print(f"   🔹 Created At: {created_at}")

            with open(CSV_LOG, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([symbol, side, qty, alpaca_side, limit_price, order_id, status, created_at])

        except requests.exceptions.HTTPError as e:
            print(f"❌ Failed to place order for {symbol}: {e}")
            print(f"🧾 Response: {r.text}")
            if "insufficient buying power" in r.text.lower():
                if side not in ("buy", "cover"):
                    print("🔁 Skipping retry logic for non-buy orders.")
                    return
            
                max_retries = 5
                attempts = 0
                next_qty = float(qty)
    
                while attempts < max_retries:
                    if next_qty > 1:
                        next_qty = int(next_qty) - 1
                    else:
                        next_qty = round(next_qty / 2, 4)
                        if next_qty < 0.01:
                            break
    
                    print(f"🔁 Retry {attempts + 1}: Attempting to buy {next_qty} shares of {symbol}")
    
                    retry_payload = {
                        "symbol": symbol,
                        "qty": next_qty,
                        "side": alpaca_side,
                        "type": "limit",
                        "limit_price": limit_price,
                        "time_in_force": "day"
                    }
    
                    retry_response = requests.post(f"{BASE_URL}/v2/orders", headers=headers, json=retry_payload)
    
                    if retry_response.ok:
                        result = retry_response.json()
                        print(f"✅ Order succeeded on retry {attempts + 1}: {result.get('qty')} shares")
    
                        with open(CSV_LOG, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([symbol, side, next_qty, alpaca_side, limit_price, result.get("id"), result.get("status"), result.get("created_at")])
                        return
                    else:
                        print(f"❌ Retry {attempts + 1} failed: {retry_response.text}")
                        attempts += 1
    
                print("❌ All retry attempts failed due to insufficient buying power.")

    except requests.exceptions.HTTPError as e:
        print(f"❌ Failed to place order for {symbol}: {e}")
        print(f"🧾 Response: {r.text}")

def determine_order_type(order):
    side = order["side"].lower()
    symbol = order["symbol"]
    qty = order["qty"]

    if side in ("buy", "sell", "short", "cover"):
        submit_order(symbol, qty, side)
    else:
        print(f"⚠️ Unknown side: {side} for {symbol}")

def main():
    print_current_holdings()

    # Auto-cancel all open orders before submitting new ones
    print("🚫 Auto-canceling all open Alpaca orders...")
    try:
        cancel_resp = requests.delete(f"{BASE_URL}/v2/orders", headers=headers)
        cancel_resp.raise_for_status()
        print("✅ All open orders cancelled.")
        print(cancel_resp.text)
    except Exception as e:
        print(f"❌ Failed to cancel open orders: {e}")
        return

    if not os.path.exists(ORDER_FILE):
        print(f"❌ Order file not found: {ORDER_FILE}")
        return

    if not TESTING_MODE:
        clock_url = f"{BASE_URL}/v2/clock"
        try:
            clock_resp = requests.get(clock_url, headers=headers)
            clock_resp.raise_for_status()
            clock_data = clock_resp.json()

            if not clock_data.get("is_open", False):
                next_open = clock_data.get("next_open")
                print(f"⏰ Market is currently closed. Next open time: {next_open}")
                return
        except Exception as e:
            print(f"❌ Failed to check market status: {e}")
            return
    else:
        print("🧪 TESTING_MODE active – skipping market hours check.")

    with open(ORDER_FILE, "r") as f:
        try:
            orders = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ JSON error in order file: {e}")
            return

    for order in orders:
        if "symbol" in order and "side" in order and "qty" in order:
            determine_order_type(order)
        else:
            print(f"⚠️ Skipping invalid order: {order}")

if __name__ == "__main__":
    main()
