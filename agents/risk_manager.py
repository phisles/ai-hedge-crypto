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

    risk_analysis = {}
    current_prices = {}

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

        # latest price
        current_price = prices_df["close"].iloc[-1]
        current_prices[ticker] = float(current_price)

        # current position value and portfolio total
        current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0)
        total_portfolio_value = (
            portfolio.get("cash", 0)
            + sum(portfolio.get("cost_basis", {}).get(t, 0)
                  for t in portfolio.get("cost_basis", {}))
        )

        # Base limit is 20% of portfolio for any single position
        position_limit = portfolio.get("cash", 0)
        remaining_position_limit = position_limit - current_position_value
        remaining_position_limit = max(0.0, remaining_position_limit)

        # final long size limit and fractional shares
        max_long_size = min(remaining_position_limit, portfolio.get("cash", 0))
        max_affordable_quantity = (max_long_size / current_price) if current_price > 0 else 0.0

        risk_analysis[ticker] = {
            "remaining_position_limit": float(remaining_position_limit),
            "current_price": float(current_price),
            "max_affordable_quantity": float(max_affordable_quantity),
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position": float(current_position_value),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
            },
        }

        # update max_shares for portfolio manager (only long)
        data.setdefault("max_shares", {})[ticker] = {
            "long": risk_analysis[ticker]["max_affordable_quantity"]
        }

        progress.update_status("risk_management_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name="risk_management_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")

    print("\n✅ FINAL max_shares from risk_management_agent:")
    for t, v in data["max_shares"].items():
        print(f"  {t}: {v}")

    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis
    state["data"]["max_shares"] = data["max_shares"]

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }