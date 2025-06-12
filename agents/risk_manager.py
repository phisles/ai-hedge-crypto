from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from tools.api import get_prices, prices_to_df
import json


##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Controls position sizing based on real-world risk factors for multiple tickers."""
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    # Initialize risk analysis for each ticker
    risk_analysis = {}
    current_prices = {}  # Store prices here to avoid redundant API calls

    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Analyzing price data")

        prices = get_prices(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
        )

        if not prices:
            progress.update_status("risk_management_agent", ticker, "Failed: No price data found")
            continue

        prices_df = prices_to_df(prices)

        progress.update_status("risk_management_agent", ticker, "Calculating position limits")

        # Calculate portfolio value
        current_price = prices_df["close"].iloc[-1]
        current_prices[ticker] = current_price  # Store the current price

        # Calculate current position value for this ticker
        current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0)

        # Calculate total portfolio value using stored prices
        total_portfolio_value = portfolio.get("cash", 0) + sum(portfolio.get("cost_basis", {}).get(t, 0) for t in portfolio.get("cost_basis", {}))

        # Base limit is 20% of portfolio for any single position
        position_limit = portfolio.get("cash", 0)

        # For existing positions, subtract current position value from limit
        remaining_position_limit = position_limit - current_position_value

        # Ensure we don't exceed available cash
        # Ensure we don't exceed available cash or short margin
        #margin_requirement = portfolio.get("margin_requirement", 0.5)
        #available_margin = portfolio.get("equity", 0) - portfolio.get("margin_used", 0)
        #max_short_position = available_margin / margin_requirement if margin_requirement > 0 else 0

        # OVERRIDE: Allow shorting based on available cash instead of margin
        available_margin = portfolio.get("cash", 0)
        max_short_position = available_margin  # Use cash as proxy for short capacity
        margin_requirement = 1.0  # Pretend no leverage required
        print(f"⚠️ OVERRIDE ENABLED — Shorting uses cash (${available_margin:.2f}) instead of margin")
        
        # Final long and short size limits
        max_long_size = min(remaining_position_limit, portfolio.get("cash", 0))
        max_short_size = min(remaining_position_limit, max_short_position)
        
        # NEW: calculate max affordable quantity
        # NEW: calculate max affordable quantity
        raw_short_qty = int(max_short_size // current_price)
        print(f"[DEBUG] FINAL SHORT QTY for {ticker} = {raw_short_qty} (raw short size = {max_short_size:.2f}, price = {current_price:.2f})")
        
        risk_analysis[ticker] = {
            "remaining_position_limit": float(remaining_position_limit),
            "current_price": float(current_price),
            "max_affordable_quantity_long": max(0, int(max_long_size // current_price)),
            "max_affordable_quantity_short": max(0, raw_short_qty),
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position": float(current_position_value),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
                "available_margin": float(available_margin),
                "margin_requirement": float(margin_requirement),
            },
        }
        if "max_shares" not in data:
            data["max_shares"] = {}
        
        data["max_shares"][ticker] = {
            "long": risk_analysis[ticker]["max_affordable_quantity_long"],
            "short": risk_analysis[ticker]["max_affordable_quantity_short"],
        }
        progress.update_status("risk_management_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name="risk_management_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")

    # Add the signal to the analyst_signals list
    print("\n✅ FINAL max_shares from risk_management_agent:")
    for t, v in data["max_shares"].items():
        print(f"  {t}: {v}")
    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis
    state["data"]["max_shares"] = data["max_shares"]

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }
