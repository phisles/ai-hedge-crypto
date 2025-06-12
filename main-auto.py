
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Style, init
from agents.portfolio_manager import portfolio_management_agent
from agents.risk_manager import risk_management_agent
from graph.state import AgentState
from utils.display import print_trading_output
from utils.analysts import ANALYST_ORDER, get_analyst_nodes
from utils.progress import progress
from llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import os
import sys
import requests
sys.path.append("/root/stock2")
# Force debug to confirm what was loaded
from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL

print("🧪 Loaded from config.py:")
print("APCA_API_KEY_ID =", APCA_API_KEY_ID)
print("APCA_API_SECRET_KEY =", APCA_API_SECRET_KEY)
print("APCA_API_BASE_URL =", APCA_API_BASE_URL)

init(autoreset=True)
TESTING_MODE = False  # Set to False in production


def fetch_alpaca_positions():
    import requests

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY
    }

    url = f"{APCA_API_BASE_URL}/v2/positions"
    print("📡 Fetching Alpaca positions...")
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    positions = {}
    for pos in data:
        symbol = pos["symbol"]
        qty = float(pos["qty"])
        cost_basis = float(pos["avg_entry_price"])
        side = pos["side"]

        positions[symbol] = {
            "long": qty if side == "long" else 0,
            "short": abs(qty) if side == "short" else 0,
            "long_cost_basis": cost_basis if side == "long" else 0.0,
            "short_cost_basis": cost_basis if side == "short" else 0.0,
            "short_margin_used": abs(float(pos["market_value"])) if side == "short" else 0.0,
        }

    return positions

def fetch_alpaca_equity():
    import requests
    api_key = APCA_API_KEY_ID
    api_secret_key = APCA_API_SECRET_KEY
    base_url = APCA_API_BASE_URL

    print("\n🔍 Attempting to fetch Alpaca equity")
    print(f"🔑 API Key: {api_key}")
    print(f"🔑 Secret Key: {api_secret_key}")
    print(f"🌐 Base URL: {base_url}")

    if not api_key or not api_secret_key:
        print(f"{Fore.RED}❌ Missing Alpaca API credentials. Check your config.py file.{Style.RESET_ALL}")
        sys.exit(1)

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret_key
    }

    try:
        print("📡 Sending request to Alpaca...")
        resp = requests.get(f"{base_url}/v2/account", headers=headers)
        print(f"🔁 Response Status: {resp.status_code}")
        print(f"📝 Response Body: {resp.text}")

        resp.raise_for_status()
        alpaca_equity = float(resp.json()["equity"])
        print(f"{Fore.YELLOW}📊 Using Alpaca account equity: ${alpaca_equity:,.2f}{Style.RESET_ALL}")
        json_data = resp.json()
        return {
            "equity": float(json_data["equity"]),
            "cash": float(json_data.get("cash", 0.0)),  # ✅ Add this line
            "margin_requirement": float(json_data.get("maintenance_margin", 0.0)),
            "margin_used": float(json_data.get("margin_used", 0.0)),
            "buying_power": float(json_data.get("buying_power", 0.0)),
            "portfolio_value": float(json_data.get("portfolio_value", 0.0)),
            "initial_margin": float(json_data.get("initial_margin", 0.0)),
        }

    except requests.exceptions.HTTPError as http_err:
        print(f"{Fore.RED}❌ HTTP error occurred: {http_err}{Style.RESET_ALL}")
    except requests.exceptions.RequestException as req_err:
        print(f"{Fore.RED}❌ Request exception: {req_err}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}⚠️ Unexpected error: {e}{Style.RESET_ALL}")

    sys.exit(1)

def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None



##### Run the Hedge Fund #####
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
):
    # Start progress tracking
    progress.start()

    try:
        # Create a new workflow if analysts are customized
        if selected_analysts:
            from pprint import pprint
            print("\n🧠 Available analyst keys from get_analyst_nodes():")
            pprint(get_analyst_nodes().keys())
            workflow = create_workflow(selected_analysts)
            agent = workflow.compile()
        else:
            agent = app

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            },
        )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        # Stop progress tracking
        progress.stop()

def get_alpaca_price(symbol):
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return float(resp.json()["trade"]["p"])



def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)

    workflow.set_entry_point("start_node")
    return workflow


if __name__ == "__main__":

    # 👇 AUTO-RUN CONFIG OVERRIDE (no questionary, no argparse)
    #LIVE
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]
    #TEST
    #tickers = ["AAPL", "NVDA", "TSLA"]
    selected_analysts = list(get_analyst_nodes().keys())
    #LIVE
    model_choice = "gpt-4o"
    #TEST
    #model_choice = "o3-mini"
    model_info = get_model_info(model_choice)
    model_provider = model_info.provider.value if model_info else "Unknown"
    show_reasoning = False

    # Set dates (auto)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=7)).strftime("%Y-%m-%d")

    # Create the workflow with selected analysts
    workflow = create_workflow(selected_analysts)
    app = workflow.compile()
    # Set the start and end dates

    # Initialize portfolio with cash amount and stock positions
    # Initialize portfolio with cash amount and stock positions
    alpaca_account = fetch_alpaca_equity()
    # 🔧 Estimate margin requirement from equity and buying power
    equity = alpaca_account["equity"]
    buying_power = alpaca_account["buying_power"]
    
    leverage = (buying_power / equity) if equity else 1
    margin_requirement = 1 / float(alpaca_account.get("multiplier", 2))
    
    alpaca_account["margin_requirement"] = margin_requirement
    print(f"📐 Estimated Margin Requirement = {margin_requirement:.2f} (Leverage = {leverage:.2f}x)")

    
    positions = fetch_alpaca_positions()
    
    manual_margin_used = sum(
        abs(pos["short"] * get_alpaca_price(symbol))
        for symbol, pos in positions.items()
        if pos["short"] > 0
    )
    long_market_value = sum(pos["long"] * get_alpaca_price(symbol) for symbol, pos in positions.items())
    short_market_value = sum(pos["short"] * get_alpaca_price(symbol) for symbol, pos in positions.items())
    
    # Use Alpaca's reported initial margin directly
    initial_margin_limit = float(alpaca_account.get("initial_margin", 0.0))
    current_position_exposure = long_market_value + abs(short_market_value)

    
    cost_basis = {
        symbol: pos["long"] * pos["long_cost_basis"] + pos["short"] * pos["short_cost_basis"]
        for symbol, pos in positions.items()
    }
    
    # ✅ Choose margin enforcement mode:
    # Strict Margin
    #remaining_position_limit = max(0, initial_margin_limit - current_position_exposure)
    
    # Loose Margin
    #remaining_position_limit = min(float(alpaca_account['cash']), float(alpaca_account['buying_power']))
    #TAKE FULL MARGIN RISK
    remaining_position_limit = float(alpaca_account['buying_power'])
    print(f"🧮 Using soft remaining position limit = ${remaining_position_limit:.2f} based on cash and buying_power")
    

    
    cost_basis = {
        symbol: pos["long"] * pos["long_cost_basis"] + pos["short"] * pos["short_cost_basis"]
        for symbol, pos in positions.items()
    }
    
    
    # Construct portfolio dict
    portfolio = {
        "cash": alpaca_account["buying_power"],
        "margin_requirement": alpaca_account["margin_requirement"],
        "margin_used": manual_margin_used,
        "positions": positions,
        "cost_basis": cost_basis,
        "realized_gains": {
            ticker: {
                "long": 0.0,
                "short": 0.0,
            } for ticker in tickers
        }
    }
    
    # Then assign this separately
    portfolio["remaining_position_limit"] = remaining_position_limit
    
    #per_symbol_limit = max(0, remaining_position_limit / len(tickers))
    #portfolio["max_position_value"] = {
    #    symbol: per_symbol_limit for symbol in tickers
    #}

    portfolio["max_position_value"] = {}  # remove artificial per-symbol caps
    
    print("\n📊 RISK-CHECKED MAX SHARES PER ASSET:")
    for symbol in tickers:
        try:
            price = get_alpaca_price(symbol)
            limit = portfolio["remaining_position_limit"]
            max_shares = max(0, int(limit // price))
            print(f"\n🧮 CALCULATING MAX SHARES for {symbol}:")
            print(f"  ➤ price = ${price:.2f}")
            print(f"  ➤ position limit = ${limit:.2f}")
            print(f"  ➤ cash = ${portfolio['cash']:.2f}")
            print(f"  ➤ margin requirement = {portfolio['margin_requirement']:.2f}")
            print(f"  ➤ margin used = ${portfolio['margin_used']:.2f}")
            
            max_long_dollars = min(portfolio["cash"], limit)
            max_short_capacity = max(0, limit)  # just use soft limit (cash or buying_power)
            
            max_shares_long = max(0, int(max_long_dollars // price))

            raw_capacity = (limit / portfolio["margin_requirement"]) - portfolio["margin_used"]
            print(f"🧪 DEBUG: Shorting calc for {symbol}")
            print(f"     limit = ${limit:.2f}")
            print(f"     margin_req = {portfolio['margin_requirement']:.2f}")
            print(f"     margin_used = ${portfolio['margin_used']:.2f}")
            print(f"     raw short capacity = ${raw_capacity:.2f}")
            max_shares_short = max(0, int(max_short_capacity // price))
            
            print(f"  ✅ LONG: max_long_dollars = ${max_long_dollars:.2f} → max_shares = {max_shares_long}")
            print(f"  ✅ SHORT: max_short_capacity = ${max_short_capacity:.2f} → max_shares = {max_shares_short}")
            portfolio.setdefault("max_shares", {})[symbol] = {
                "long": max_shares_long,
                "short": max_shares_short,
            }
        except Exception as e:
            print(f"⚠️ Failed to compute max shares for {symbol}: {e}")

    # ✅ Print snapshot of holdings for confirmation
    print("\n🔎 RAW VALUES FROM ALPACA ACCOUNT FETCH:")
    print(f"  ➤ equity                 → {alpaca_account['equity']}")
    print(f"  ➤ maintenance_margin     → {alpaca_account['margin_requirement']}")
    print(f"  ➤ margin_used            → {alpaca_account['margin_used']}")
    print(f"  ➤ buying_power           → {alpaca_account['buying_power']}")
    print(f"  ➤ portfolio_value        → {alpaca_account['portfolio_value']}")
    print(f"  ➤ initial_margin_limit   → {initial_margin_limit:.2f}")
    print(f"  ➤ long_market_value      → {long_market_value:.2f}")
    print(f"  ➤ short_market_value     → {short_market_value:.2f}")
    print(f"  ➤ current_position_value → {current_position_exposure:.2f}")
    print(f"  ➤ remaining_position_limit → {remaining_position_limit:.2f}")
    
    print("\n📦 ASSIGNING TO PORTFOLIO DICT:")
    print(f"  ➤ portfolio['cash']                = {portfolio['cash']}")
    print(f"  ➤ portfolio['margin_requirement']  = {portfolio['margin_requirement']}")
    print(f"  ➤ portfolio['margin_used']         = {portfolio['margin_used']}")
    print(f"  ➤ portfolio['remaining_position_limit'] = {portfolio['remaining_position_limit']}")
    print(f"  ➤ portfolio['cost_basis']          = {json.dumps(portfolio['cost_basis'], indent=2)}")
    print(f"  ➤ portfolio['max_position_value']  = {json.dumps(portfolio['max_position_value'], indent=2)}")
    
    for symbol, pos in portfolio["positions"].items():
        long_qty = pos["long"]
        short_qty = pos["short"]
        if long_qty > 0 or short_qty > 0:
            print(f"🔹 {symbol}: ", end="")
            if long_qty > 0:
                print(f"Long {long_qty} @ ${pos['long_cost_basis']:.2f}", end="")
            if short_qty > 0:
                if long_qty > 0:
                    print(" | ", end="")
                print(f"Short {short_qty} @ ${pos['short_cost_basis']:.2f} (Margin Used: ${pos['short_margin_used']:.2f})", end="")
            print()
    
    if not TESTING_MODE:
        clock_url = f"{APCA_API_BASE_URL}/v2/clock"
        try:
            print("\n⏰ Checking market hours...")
            clock_resp = requests.get(clock_url, headers={
                "APCA-API-KEY-ID": APCA_API_KEY_ID,
                "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY
            })
            clock_resp.raise_for_status()
            clock_data = clock_resp.json()
    
            if not clock_data.get("is_open", False):
                next_open = clock_data.get("next_open")
                print(f"🚫 Market is currently closed. Next open time: {next_open}")
                sys.exit(0)
            else:
                print("✅ Market is open.")
        except Exception as e:
            print(f"❌ Failed to check market status: {e}")
            sys.exit(1)
    else:
        print("🧪 TESTING_MODE active – skipping market hours check.")
    # Run the hedge fund
    # Allocate shares using total available funds (remaining_position_limit)
    available_funds = portfolio["remaining_position_limit"]
    buy_plan = {}
    
    print("\n🧮 ALLOCATING FUNDS ACROSS TOP-RANKED ASSETS:")
    for symbol in tickers:
        try:
            price = get_alpaca_price(symbol)
            max_shares = int(available_funds // price)
            if max_shares > 0:
                buy_plan[symbol] = max_shares
                used = max_shares * price
                available_funds -= used
                print(f"  ✅ {symbol}: max_shares={max_shares}, used=${used:.2f}, remaining=${available_funds:.2f}")
        except Exception as e:
            print(f"⚠️ Failed to fetch price for {symbol}: {e}")
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        selected_analysts=selected_analysts,
        model_name=model_choice,
        model_provider=model_provider,
    )
    print_trading_output(result)
    # ✅ Save Alpaca order info to JSON
    # Ensure decisions are parsed
    # Ensure decisions are parsed
    decisions = result["decisions"]
    print("\n🧪 RAW DECISIONS:")
    for k, v in decisions.items():
        print(f"  {k}: action={v['action']}, qty={v['quantity']}, confidence={v['confidence']:.2f}")
    print("\n💰 ACCOUNT STATUS FROM ALPACA:")
    print(f"  ➤ Buying Power       = ${alpaca_account['buying_power']:.2f}")
    print(f"  ➤ Equity             = ${alpaca_account['equity']:.2f}")
    print(f"  ➤ Portfolio Value    = ${alpaca_account['portfolio_value']:.2f}")
    print(f"  ➤ Margin Used        = ${alpaca_account['margin_used']:.2f}")
    print(f"  ➤ Maintenance Margin = ${alpaca_account['margin_requirement']:.2f}")

    if isinstance(decisions, str):
        decisions = parse_hedge_fund_response(decisions)

    # ✅ Save Alpaca order info to JSON
    orders_to_save = []

    estimated_total_cost = 0.0
    orders_to_save = []
    
    if isinstance(decisions, dict):
        # Convert decisions to a working list
        adjusted_orders = []
        for symbol, details in decisions.items():
            action = details.get("action", "").lower()
            qty = details.get("quantity", 0)
            if action in ("buy", "short", "sell", "cover") and qty > 0:
                side = "buy" if action == "cover" else action
                adjusted_orders.append({"symbol": symbol, "side": side, "qty": qty})
    
        # Reduce loop
        # Fetch prices and attach confidence for buy/short orders
        candidates = []
        total_confidence = 0.0
        for symbol, details in decisions.items():
            action = details.get("action", "").lower()
            qty = details.get("quantity", 0)
            confidence = details.get("confidence", 0.0)
            if action in ("buy", "short", "sell", "cover") and qty > 0:
                try:
                    price = get_alpaca_price(symbol)
                except Exception as e:
                    print(f"⚠️ Price fetch failed for {symbol}: {e}")
                    continue
                candidates.append({
                    "symbol": symbol,
                    "side": "buy" if action == "cover" else action,
                    "price": price,
                    "confidence": confidence,
                })
                print(f"📥 Candidate added: {symbol} → side={action} → mapped_side={'buy' if action == 'cover' else action}, price=${price:.2f}, confidence={confidence}")
                total_confidence += confidence
        
        available_funds = alpaca_account["buying_power"]
        adjusted_orders = []
        
        if total_confidence == 0:
            print("⚠️ Total confidence is zero; cannot allocate proportionally.")
        else:
            for c in candidates:
                share_budget = available_funds * (c["confidence"] / total_confidence)
                max_qty = int(share_budget // c["price"])
                print(f"📊 Allocating {c['symbol']} → side={c['side']}, qty={max_qty}, share_budget=${share_budget:.2f}, price=${c['price']:.2f}")
                if max_qty > 0:
                    adjusted_orders.append({
                        "symbol": c["symbol"],
                        "side": c["side"],
                        "qty": max_qty,
                        "price": c["price"],
                        "confidence": c["confidence"],
                    })
        
        # Optional: print allocation debug info
        # Reduction loop to ensure total cost fits within buying power
        while True:
            total_allocated = sum(o["qty"] * o["price"] for o in adjusted_orders)
            if total_allocated <= available_funds:
                print(f"✅ Budget fits: total_allocated=${total_allocated:.2f} within available_funds=${available_funds:.2f}")
                break

            print(f"⚠️ Budget exceeded: total_allocated=${total_allocated:.2f}, reducing quantities...")

            orders_with_qty = [o for o in adjusted_orders if o["qty"] > 0]
            if not orders_with_qty:
                print("⚠️ All quantities zero, cannot reduce further.")
                break

            orders_with_qty.sort(key=lambda x: x["confidence"])

            order_to_reduce = orders_with_qty[0]
            order_to_reduce["qty"] -= 1
            print(f"🔽 Reduced {order_to_reduce['symbol']} qty to {order_to_reduce['qty']} (confidence: {order_to_reduce['confidence']:.2f})")

        # Final print of allocations
        total_allocated = sum(o["qty"] * o["price"] for o in adjusted_orders)
        print(f"🧮 Allocated orders total cost: ${total_allocated:.2f} within buying power ${available_funds:.2f}")

        print(f"\n🧮 TOTAL ESTIMATED TRADE COST: ${total_allocated:,.2f}")
        orders_to_save = []
        for order in adjusted_orders:
            if order["qty"] > 0:
                print(f"  ➤ {order['symbol']} → {order['side'].upper()} {order['qty']} @ ${order['price']:.2f} = ${order['qty'] * order['price']:.2f}")
                orders_to_save.append({
                    "symbol": order["symbol"],
                    "side": "buy" if order["side"] == "cover" else order["side"],
                    "qty": order["qty"]
                })
                print(f"✅ FINAL ORDER → {order['symbol']} {order['side'].upper()} {order['qty']} @ ${order['price']:.2f}")


    elif isinstance(decisions, list):
        for item in decisions:
            action = item.get("action", "").lower()
            qty = item.get("quantity", 0)
            symbol = item.get("symbol", "UNKNOWN")
            print(f"🔍 Checking {symbol}: action={action}, quantity={qty}")

            if action in ("buy", "sell", "short", "cover") and qty > 0:
                side = "buy" if action == "cover" else action
                if action == "cover":
                    print(f"🔁 Converted COVER to BUY for {symbol}")
                orders_to_save.append({
                    "symbol": symbol,
                    "side": side,
                    "qty": qty
                })
            else:
                print(f"⚠️ Skipped {symbol} - not a valid order (action or qty)")
    else:
        print(f"⚠️ ERROR: Unknown format for decisions: {type(decisions)}")

    output_path = os.path.join(os.path.dirname(__file__), "order-data", "alpaca_order_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(orders_to_save, f, indent=2)

    print(f"\n📝 Saved {len(orders_to_save)} Alpaca orders to {output_path}")
