from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import json

from tools.api import get_financial_metrics, get_market_cap, search_line_items


##### Crypto Valuation Agent #####
def valuation_agent(state: AgentState):
    """
    Performs detailed valuation analysis for crypto assets using a mix of:
      - Traditional on-chain valuation (NVT, address growth, developer activity)
      - Classical DCF and Owner Earnings methods where applicable
    """
    data = state["data"]
    end_date = data.get("end_date")
    tickers = data.get("tickers", [])

    valuation_analysis = {}

    for ticker in tickers:
        progress.update_status("valuation_agent", ticker, "Fetching on-chain metrics")
        # On-chain metrics via financial API
        crypto_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
        )
        if not crypto_metrics:
            progress.update_status("valuation_agent", ticker, "Failed: No on-chain metrics found")
            continue
        cm = crypto_metrics[0]

        market_cap_onchain = cm.get("market_cap", 0.0)
        total_volume = cm.get("total_volume", 0.0)
        addr_growth = cm.get("active_addresses_24h")
        dev_activity = cm.get("developer_activity")

        # Traditional financial metrics fallback
        progress.update_status("valuation_agent", ticker, "Fetching financial line items")
        financial_line_items = search_line_items(
            ticker=ticker,
            line_items=[
                "free_cash_flow",
                "net_income",
                "depreciation_and_amortization",
                "capital_expenditure",
                "working_capital",
            ],
            end_date=end_date,
            limit=2,
        )

        # Compute Owner Earnings & DCF if line items exist
        owner_earnings_value = 0.0
        dcf_value = 0.0
        if len(financial_line_items) >= 2:
            cur = financial_line_items[0]
            prev = financial_line_items[1]
            wc_change = cur.working_capital - prev.working_capital
            owner_earnings_value = calculate_owner_earnings_value(
                net_income=cur.net_income,
                depreciation=cur.depreciation_and_amortization,
                capex=cur.capital_expenditure,
                working_capital_change=wc_change,
                growth_rate=cm.get("earnings_growth", 0.0),
                required_return=0.15,
                margin_of_safety=0.25,
            )
            dcf_value = calculate_intrinsic_value(
                free_cash_flow=cur.free_cash_flow or 0.0,
                growth_rate=cm.get("earnings_growth", 0.0),
                discount_rate=0.10,
                terminal_growth_rate=0.03,
                num_years=5,
            )

        # Combine crypto and traditional valuations
        # 50% weight on NVT, 25% on Owner Earnings, 25% on DCF
        nvt_ratio = None
        if total_volume > 0:
            nvt_ratio = market_cap_onchain / total_volume
        # Score each component
        nvt_score = 0.0
        if nvt_ratio is not None:
            if nvt_ratio < 20:
                nvt_score = 1.0
            elif nvt_ratio > 80:
                nvt_score = -1.0
        oe_score = 0.0 if owner_earnings_value == 0 or market_cap_onchain == 0 else (owner_earnings_value - market_cap_onchain) / market_cap_onchain
        dcf_score = 0.0 if dcf_value == 0 or market_cap_onchain == 0 else (dcf_value - market_cap_onchain) / market_cap_onchain

        # Developer activity score
        dev_score = 0.0
        if dev_activity is not None:
            if dev_activity > 60000:
                dev_score = 0.25
            elif dev_activity > 30000:
                dev_score = 0.10
            elif dev_activity < 10000:
                dev_score = -0.10

        combined_score = (
            (nvt_score) * 0.40
            + oe_score * 0.20
            + dcf_score * 0.20
            + dev_score * 0.20
        )

        if combined_score > 0.15:
            signal = "bullish"
        elif combined_score < -0.15:
            signal = "bearish"
        else:
            signal = "neutral"

        # Confidence is abs(combined_score)/0.30 capped at 100%
        confidence = min(abs(combined_score) / 0.30 * 100, 100)
        confidence = round(confidence)

        # Build reasoning
        reasoning = {
            "nvt_ratio": f"{nvt_ratio:.1f}" if nvt_ratio is not None else "N/A",
            "owner_earnings_value": owner_earnings_value,
            "dcf_value": dcf_value,
            "thresholds": "NVT<20 bullish, >80 bearish; gap>15% bullish/bearish",
            "active_addresses_24h": addr_growth,
            "developer_activity": dev_activity,
        }

        valuation_analysis[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }
        progress.update_status("valuation_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(valuation_analysis),
        name="valuation_agent",
    )
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(valuation_analysis, "Valuation Analysis Agent")
    state["data"]["analyst_signals"]["valuation_agent"] = valuation_analysis
    return {"messages": [message], "data": data}


def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5,
) -> float:
    if not all(isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]):
        return 0.0
    owner_earnings = net_income + depreciation - capex - working_capital_change
    if owner_earnings <= 0:
        return 0.0
    future_values = []
    for year in range(1, num_years + 1):
        fv = owner_earnings * (1 + growth_rate) ** year
        dv = fv / (1 + required_return) ** year
        future_values.append(dv)
    terminal_growth = min(growth_rate, 0.03)
    terminal = (future_values[-1] * (1 + terminal_growth)) / (required_return - terminal_growth)
    terminal_discounted = terminal / (1 + required_return) ** num_years
    intrinsic = sum(future_values) + terminal_discounted
    return intrinsic * (1 - margin_of_safety)


def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    cash_flows = [free_cash_flow * (1 + growth_rate) ** i for i in range(1, num_years + 1)]
    pv = [cf / (1 + discount_rate) ** idx for idx, cf in enumerate(cash_flows, start=1)]
    terminal = cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    terminal_pv = terminal / (1 + discount_rate) ** num_years
    return sum(pv) + terminal_pv


def calculate_working_capital_change(
    current_working_capital: float,
    previous_working_capital: float,
) -> float:
    return current_working_capital - previous_working_capital
