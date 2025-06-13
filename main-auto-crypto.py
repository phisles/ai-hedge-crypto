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
import base64
import time
import hashlib
import requests
import hmac

# Load environment variables
load_dotenv()

# Ensure local modules path
sys.path.append("/root/stock2")

from config2 import GEM_API_KEY, GEM_API_SECRET  # Use your Gemini keys

init(autoreset=True)
TESTING_MODE = False  # Set to False in production


def get_gemini_portfolio():
    url = "https://api.gemini.com/v1/notionalbalances/usd"
    payload = {
        "request": "/v1/notionalbalances/usd",
        "nonce": int(time.time() * 1000)
    }
    encoded_payload = base64.b64encode(json.dumps(payload).encode()).decode()
    signature = hmac.new(GEM_API_SECRET.encode(), encoded_payload.encode(), hashlib.sha384).hexdigest()

    headers = {
        "X-GEMINI-APIKEY": GEM_API_KEY,
        "X-GEMINI-PAYLOAD": encoded_payload,
        "X-GEMINI-SIGNATURE": signature,
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    total_value = sum(float(item["amountNotional"]) for item in data if "amountNotional" in item)
    assets = [
        {
            "currency": item["currency"],
            "amountNotional": float(item["amountNotional"]),
            "available": float(item["available"])
        }
        for item in data if "amountNotional" in item
    ]

    return {"total_balance": total_value, "assets": assets}


def get_gemini_price(symbol):
    pair = symbol.replace("/", "").lower()
    url = f"https://api.gemini.com/v1/pubticker/{pair}"
    resp = requests.get(url)
    resp.raise_for_status()
    return float(resp.json()["ask"])


def parse_hedge_fund_response(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}; response: {response}")
        return None
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None


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
    progress.start()
    try:
        if selected_analysts:
            from pprint import pprint
            print("\n🧠 Available analyst keys:")
            pprint(get_analyst_nodes().keys())
            workflow = create_workflow(selected_analysts)
            agent = workflow.compile()
        else:
            workflow = create_workflow()
            agent = workflow.compile()

        final_state = agent.invoke(
            {
                "messages": [HumanMessage(content="Make trading decisions based on the provided data.")],
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
            }
        )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }

    finally:
        progress.stop()


def start(state: AgentState):
    return state


from utils.analysts import ANALYST_CONFIG

def create_workflow(selected_analysts=None):
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)
    analyst_nodes = get_analyst_nodes()

    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())

    # Enforce order from ANALYST_CONFIG["order"]
    ordered_keys = sorted(selected_analysts, key=lambda x: ANALYST_CONFIG[x]["order"])

    for key in ordered_keys:
        name, func = analyst_nodes[key]
        workflow.add_node(name, func)
        workflow.add_edge("start_node", name)

    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)

    for key in ordered_keys:
        name = analyst_nodes[key][0]
        workflow.add_edge(name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)
    workflow.set_entry_point("start_node")
    return workflow


if __name__ == "__main__":
    # Configuration
    #tickers = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "DOT/USD"]
    tickers = ["BTC/USD", "ETH/USD"]
    
    
    #LIVE
    from utils.analysts import ANALYST_CONFIG

    selected_analysts = [
        key for key, _ in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])
]
    
    #TEST
    #selected_analysts = ["ben_graham"]

    model_choice = "gpt-4.1-nano"
    model_info = get_model_info(model_choice)
    model_provider = model_info.provider.value
    show_reasoning = False

    # Date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=7)).strftime("%Y-%m-%d")

    # Fetch Gemini portfolio once
    portfolio_data = get_gemini_portfolio()
    assets = portfolio_data.get("assets", [])
    total_value = portfolio_data.get("total_balance", 0.0)
    cash = next((item["amountNotional"] for item in assets if item["currency"] == "USD"), 0.0)

    # Build positions dict (long only)
    positions = {
        f"{item['currency'].upper()}/USD": item["available"]
        for item in assets
        if item["currency"].upper() != "USD" and item["available"] > 0
    }

    # Construct portfolio
    portfolio = {
        "cash": cash,
        "positions": {symbol: {"long": qty, "long_cost_basis": 0.0} for symbol, qty in positions.items()},
        "cost_basis": {symbol: 0.0 for symbol in positions},
        "realized_gains": {ticker: 0.0 for ticker in tickers},
        "remaining_position_limit": cash,
        "max_position_value": {}
    }

    # Display summary
    print(f"📥 Total Portfolio Value: ${total_value:.2f}")
    print(f"💰 Cash Available: ${cash:.2f}")
    print("\n📊 Current Positions:")
    for sym, qty in positions.items():
        print(f"  🔹 {sym}: {qty}")

    # Risk-check max shares
    print("\n📊 RISK-CHECKED MAX SHARES PER ASSET:")
    for symbol in tickers:
        try:
            price = get_gemini_price(symbol)
            max_shares = round(portfolio["remaining_position_limit"] / price, 8)
            print(f"  ✅ {symbol}: max_shares={max_shares}")
            portfolio.setdefault("max_shares", {})[symbol] = max_shares
        except Exception as e:
            print(f"⚠️ Failed to fetch price for {symbol}: {e}")

    # Allocate funds across top assets
    available_funds = portfolio["remaining_position_limit"]
    print("\n🧮 ALLOCATING FUNDS ACROSS TOP-RANKED ASSETS:")
    buy_plan = {}
    for symbol in tickers:
        try:
            price = get_gemini_price(symbol)
            qty = round(available_funds / price, 8)
            if qty > 0:
                buy_plan[symbol] = qty
                used = qty * price
                available_funds -= used
                print(f"  ✅ {symbol}: qty={qty}, used=${used:.2f}, remaining=${available_funds:.2f}")
        except Exception as e:
            print(f"⚠️ Price fetch failed for {symbol}: {e}")

    # Run hedge fund workflow
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

    # Parse decisions
    decisions = result.get("decisions")
    if isinstance(decisions, str):
        decisions = parse_hedge_fund_response(decisions)

    # Print and save orders
    print("\n🧪 RAW DECISIONS:")
    orders_to_save = []
    if isinstance(decisions, dict):
        candidates = []
        total_confidence = 0.0
        for symbol, details in decisions.items():
            action = details.get("action", "").lower()
            qty = details.get("quantity", 0)
            confidence = details.get("confidence", 0.0)
            if action in ("buy", "sell") and qty > 0:
                price = get_gemini_price(symbol)
                candidates.append({"symbol": symbol, "side": action, "price": price, "confidence": confidence})
                total_confidence += confidence
                print(f"📥 Candidate: {symbol} side={action} qty={qty} price=${price:.2f} conf={confidence:.2f}")
        if total_confidence == 0:
            print("⚠️ Total confidence zero; skipping allocation.")
        else:
            available = cash
            adjusted = []
            for c in candidates:
                budget = available * (c["confidence"] / total_confidence)
                qty = int(budget // c["price"])
                if qty > 0:
                    adjusted.append({"symbol": c["symbol"], "side": c["side"], "qty": qty, "price": c["price"]})
                    print(f"📊 Allocating {c['symbol']} qty={qty} budget=${budget:.2f}")
            while sum(o["qty"] * o["price"] for o in adjusted) > cash:
                adjusted.sort(key=lambda x: x["qty"])
                adj = adjusted[0]
                adj["qty"] -= 1
            total_allocated = sum(o["qty"] * o["price"] for o in adjusted)
            print(f"🧮 Total Allocated: ${total_allocated:.2f}")
            for o in adjusted:
                print(f"  ➤ {o['symbol']} {o['side'].upper()} {o['qty']} @ ${o['price']:.2f}")
                orders_to_save.append({"symbol": o["symbol"], "side": o["side"], "qty": o["qty"]})
    elif isinstance(decisions, list):
        for item in decisions:
            action = item.get("action", "").lower()
            qty = item.get("quantity", 0)
            symbol = item.get("symbol", "")
            if action in ("buy", "sell") and qty > 0:
                orders_to_save.append({"symbol": symbol, "side": action, "qty": qty})

    # Save orders to file
    output_path = os.path.join(os.path.dirname(__file__), "order-data", "gemini_order_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(orders_to_save, f, indent=2)
    print(f"📝 Saved {len(orders_to_save)} orders to {output_path}")
