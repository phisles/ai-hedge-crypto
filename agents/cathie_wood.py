from graph.state import AgentState, show_agent_reasoning
from tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm

class CathieWoodSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def cathie_wood_agent(state: AgentState):
    """
    Analyzes stocks and crypto using Cathie Wood's investing principles.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    cw_analysis = {}

    for ticker in tickers:
        progress.update_status("cathie_wood_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date)

        progress.update_status("cathie_wood_agent", ticker, "Gathering financial line items")
        if ticker.upper().endswith(("/USD", "-USD")):
            financial_line_items = []
        else:
            financial_line_items = search_line_items(
                ticker,
                [
                    "revenue",
                    "gross_margin",
                    "operating_margin",
                    "debt_to_equity",
                    "free_cash_flow",
                    "total_assets",
                    "total_liabilities",
                    "dividends_and_other_cash_distributions",
                    "outstanding_shares",
                    "research_and_development",
                    "capital_expenditure",
                    "operating_expense",
                ],
                end_date,
                period="annual",
                limit=5
            )

        progress.update_status("cathie_wood_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

        progress.update_status("cathie_wood_agent", ticker, "Analyzing disruptive potential")
        disruptive_analysis = analyze_disruptive_potential(metrics, financial_line_items)

        progress.update_status("cathie_wood_agent", ticker, "Analyzing innovation-driven growth")
        innovation_analysis = analyze_innovation_growth(metrics, financial_line_items)

        progress.update_status("cathie_wood_agent", ticker, "Calculating valuation & high-growth scenario")
        valuation_analysis = analyze_cathie_wood_valuation(financial_line_items, market_cap)

        total_score = disruptive_analysis["score"] + innovation_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15

        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "disruptive_analysis": disruptive_analysis,
            "innovation_analysis": innovation_analysis,
            "valuation_analysis": valuation_analysis
        }

        progress.update_status("cathie_wood_agent", ticker, "Generating Cathie Wood analysis")
        cw_output = generate_cathie_wood_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        cw_analysis[ticker] = {
            "signal": cw_output.signal,
            "confidence": cw_output.confidence,
            "reasoning": cw_output.reasoning
        }

        progress.update_status("cathie_wood_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(cw_analysis),
        name="cathie_wood_agent"
    )

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(cw_analysis, "Cathie Wood Agent")

    state["data"]["analyst_signals"]["cathie_wood_agent"] = cw_analysis

    return {
        "messages": [message],
        "data": state["data"]
    }


def analyze_disruptive_potential(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze disruptive potential for equities and crypto.
    """
    # crypto-specific branch
    if isinstance(metrics, list) and metrics and metrics[0].get("price_change_pct_30d") is not None:
        latest = metrics[0]
        score = 0
        details = []
        if latest["price_change_pct_30d"] > 0.20:
            score += 3
            details.append(f"Strong 30d price momentum: {latest['price_change_pct_30d']:.2%}")
        elif latest["price_change_pct_30d"] > 0.10:
            score += 2
            details.append(f"Moderate 30d price momentum: {latest['price_change_pct_30d']:.2%}")
        if latest.get("developer_stars", 0) > 50000:
            score += 2
            details.append(f"High developer activity: {latest['developer_stars']} stars")
        normalized_score = (score / 5) * 5
        return {
            "score": normalized_score,
            "details": "; ".join(details),
            "raw_score": score,
            "max_score": 5
        }

    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze disruptive potential"
        }

    score = 0
    details = []

    # 1. Revenue Growth Analysis
    revenues = [item.revenue for item in financial_line_items if getattr(item, "revenue", None) is not None]
    if len(revenues) >= 3:
        growth_rates = []
        for i in range(len(revenues)-1):
            prev, curr = revenues[i], revenues[i+1]
            if prev != 0:
                growth_rates.append((curr - prev) / abs(prev))
        if len(growth_rates) >= 2 and growth_rates[-1] > growth_rates[0]:
            score += 2
            details.append(f"Revenue growth is accelerating: {(growth_rates[-1]*100):.1f}% vs {(growth_rates[0]*100):.1f}%")
        latest_growth = growth_rates[-1] if growth_rates else 0
        if latest_growth > 1.0:
            score += 3
            details.append(f"Exceptional revenue growth: {(latest_growth*100):.1f}%")
        elif latest_growth > 0.5:
            score += 2
            details.append(f"Strong revenue growth: {(latest_growth*100):.1f}%")
        elif latest_growth > 0.2:
            score += 1
            details.append(f"Moderate revenue growth: {(latest_growth*100):.1f}%")
    else:
        details.append("Insufficient revenue data for growth analysis")

    # 2. Gross Margin Analysis
    gross_margins = [item.gross_margin for item in financial_line_items if getattr(item, "gross_margin", None) is not None]
    if len(gross_margins) >= 2:
        margin_trend = gross_margins[-1] - gross_margins[0]
        if margin_trend > 0.05:
            score += 2
            details.append(f"Expanding gross margins: +{(margin_trend*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(f"Slightly improving gross margins: +{(margin_trend*100):.1f}%")
        if gross_margins[-1] > 0.50:
            score += 2
            details.append(f"High gross margin: {(gross_margins[-1]*100):.1f}%")
    else:
        details.append("Insufficient gross margin data")

    # 3. Operating Leverage Analysis
    operating_expenses = [item.operating_expense for item in financial_line_items if getattr(item, "operating_expense", None) is not None]
    if len(revenues) >= 2 and len(operating_expenses) >= 2:
        rev_growth = (revenues[-1] - revenues[0]) / abs(revenues[0]) if revenues[0] != 0 else 0
        opex_growth = (operating_expenses[-1] - operating_expenses[0]) / abs(operating_expenses[0]) if operating_expenses[0] != 0 else 0
        if rev_growth > opex_growth:
            score += 2
            details.append("Positive operating leverage: Revenue growing faster than expenses")
    else:
        details.append("Insufficient data for operating leverage analysis")

    # 4. R&D Investment Analysis
    rd_expenses = [item.research_and_development for item in financial_line_items if getattr(item, "research_and_development", None) is not None]
    if rd_expenses and revenues:
        rd_intensity = rd_expenses[-1] / revenues[-1] if revenues[-1] != 0 else 0
        if rd_intensity > 0.15:
            score += 3
            details.append(f"High R&D investment: {(rd_intensity*100):.1f}% of revenue")
        elif rd_intensity > 0.08:
            score += 2
            details.append(f"Moderate R&D investment: {(rd_intensity*100):.1f}% of revenue")
        elif rd_intensity > 0.05:
            score += 1
            details.append(f"Some R&D investment: {(rd_intensity*100):.1f}% of revenue")
    else:
        details.append("No R&D data available")

    normalized_score = (score / 12) * 5
    return {
        "score": normalized_score,
        "details": "; ".join(details),
        "raw_score": score,
        "max_score": 12
    }


def analyze_innovation_growth(metrics: list, financial_line_items: list) -> dict:
    """
    Evaluate the company's commitment to innovation and potential for exponential growth.
    Analyzes multiple dimensions:
    1. R&D Investment Trends - measures commitment to innovation
    2. Free Cash Flow Generation - indicates ability to fund innovation
    3. Operating Efficiency - shows scalability of innovation
    4. Capital Allocation - reveals innovation-focused management
    5. Growth Reinvestment - demonstrates commitment to future growth
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze innovation-driven growth"
        }

    # 1. R&D Investment Trends
    rd_expenses = [
        item.research_and_development
        for item in financial_line_items
        if hasattr(item, "research_and_development") and item.research_and_development
    ]
    revenues = [item.revenue for item in financial_line_items if item.revenue]

    if rd_expenses and revenues and len(rd_expenses) >= 2:
        # Check R&D growth rate
        rd_growth = (rd_expenses[-1] - rd_expenses[0]) / abs(rd_expenses[0]) if rd_expenses[0] != 0 else 0
        if rd_growth > 0.5:  # 50% growth in R&D
            score += 3
            details.append(f"Strong R&D investment growth: +{(rd_growth*100):.1f}%")
        elif rd_growth > 0.2:
            score += 2
            details.append(f"Moderate R&D investment growth: +{(rd_growth*100):.1f}%")

        # Check R&D intensity trend
        rd_intensity_start = rd_expenses[0] / revenues[0]
        rd_intensity_end = rd_expenses[-1] / revenues[-1]
        if rd_intensity_end > rd_intensity_start:
            score += 2
            details.append(f"Increasing R&D intensity: {(rd_intensity_end*100):.1f}% vs {(rd_intensity_start*100):.1f}%")
    else:
        details.append("Insufficient R&D data for trend analysis")

    # 2. Free Cash Flow Analysis
    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow]
    if fcf_vals and len(fcf_vals) >= 2:
        # Check FCF growth and consistency
        fcf_growth = (fcf_vals[-1] - fcf_vals[0]) / abs(fcf_vals[0])
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)

        if fcf_growth > 0.3 and positive_fcf_count == len(fcf_vals):
            score += 3
            details.append("Strong and consistent FCF growth, excellent innovation funding capacity")
        elif positive_fcf_count >= len(fcf_vals) * 0.75:
            score += 2
            details.append("Consistent positive FCF, good innovation funding capacity")
        elif positive_fcf_count > len(fcf_vals) * 0.5:
            score += 1
            details.append("Moderately consistent FCF, adequate innovation funding capacity")
    else:
        details.append("Insufficient FCF data for analysis")

    # 3. Operating Efficiency Analysis
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin]
    if op_margin_vals and len(op_margin_vals) >= 2:
        # Check margin improvement
        margin_trend = op_margin_vals[-1] - op_margin_vals[0]

        if op_margin_vals[-1] > 0.15 and margin_trend > 0:
            score += 3
            details.append(f"Strong and improving operating margin: {(op_margin_vals[-1]*100):.1f}%")
        elif op_margin_vals[-1] > 0.10:
            score += 2
            details.append(f"Healthy operating margin: {(op_margin_vals[-1]*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append("Improving operating efficiency")
    else:
        details.append("Insufficient operating margin data")

    # 4. Capital Allocation Analysis
    capex = [item.capital_expenditure for item in financial_line_items if hasattr(item, 'capital_expenditure') and item.capital_expenditure]
    if capex and revenues and len(capex) >= 2:
        capex_intensity = abs(capex[-1]) / revenues[-1]
        capex_growth = (abs(capex[-1]) - abs(capex[0])) / abs(capex[0]) if capex[0] != 0 else 0

        if capex_intensity > 0.10 and capex_growth > 0.2:
            score += 2
            details.append("Strong investment in growth infrastructure")
        elif capex_intensity > 0.05:
            score += 1
            details.append("Moderate investment in growth infrastructure")
    else:
        details.append("Insufficient CAPEX data")

    # 5. Growth Reinvestment Analysis
    dividends = [item.dividends_and_other_cash_distributions for item in financial_line_items if hasattr(item, 'dividends_and_other_cash_distributions') and item.dividends_and_other_cash_distributions]
    if dividends and fcf_vals:
        # Check if company prioritizes reinvestment over dividends
        latest_payout_ratio = dividends[-1] / fcf_vals[-1] if fcf_vals[-1] != 0 else 1
        if latest_payout_ratio < 0.2:  # Low dividend payout ratio suggests reinvestment focus
            score += 2
            details.append("Strong focus on reinvestment over dividends")
        elif latest_payout_ratio < 0.4:
            score += 1
            details.append("Moderate focus on reinvestment over dividends")
    else:
        details.append("Insufficient dividend data")

    # Normalize score to be out of 5
    max_possible_score = 15  # Sum of all possible points
    normalized_score = (score / max_possible_score) * 5

    return {
        "score": normalized_score,
        "details": "; ".join(details),
        "raw_score": score,
        "max_score": max_possible_score
    }


def analyze_cathie_wood_valuation(financial_line_items: list, market_cap: float) -> dict:
    """
    Cathie Wood often focuses on long-term exponential growth potential. We can do
    a simplified approach looking for a large total addressable market (TAM) and the
    company's ability to capture a sizable portion.
    """
    if not financial_line_items or market_cap is None:
        return {
            "score": 0,
            "details": "Insufficient data for valuation"
        }

    latest = financial_line_items[-1]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0

    if fcf <= 0:
        return {
            "score": 0,
            "details": f"No positive FCF for valuation; FCF = {fcf}",
            "intrinsic_value": None
        }

    # Instead of a standard DCF, let's assume a higher growth rate for an innovative company.
    # Example values:
    growth_rate = 0.20  # 20% annual growth
    discount_rate = 0.15
    terminal_multiple = 25
    projection_years = 5

    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    # Terminal Value
    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) \
                     / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value

    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    score = 0
    if margin_of_safety > 0.5:
        score += 3
    elif margin_of_safety > 0.2:
        score += 1

    details = [
        f"Calculated intrinsic value: ~{intrinsic_value:,.2f}",
        f"Market cap: ~{market_cap:,.2f}",
        f"Margin of safety: {margin_of_safety:.2%}"
    ]

    return {
        "score": score,
        "details": "; ".join(details),
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": margin_of_safety
    }


def generate_cathie_wood_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> CathieWoodSignal:
    """
    Generates crypto investment decisions in the style of Cathie Wood.
    """
    signal = analysis_data[ticker]["signal"]  # Extract signal for this asset

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Cathie Wood AI agent, making investment decisions on cryptocurrency assets using her principles:

1. Seek projects leveraging disruptive blockchain or decentralized technologies.
2. Emphasize exponential network adoption and on-chain activity metrics.
3. Focus on digital assets with strong developer activity, community growth, and large TAM in Web3.
4. Consider multi-year time horizons for token adoption and network growth.
5. Accept higher volatility inherent to crypto in pursuit of high returns.
6. Evaluate on-chain fundamentals: transaction volume, active addresses, staking metrics, TVL, NVT ratio.
7. Use a growth-biased valuation approach based on network metrics rather than corporate financials.

Rules:
- Identify disruptive protocol features or novel consensus mechanisms.
- Evaluate potential for exponential user adoption and network effects.
- Check if the tokenomics incentivize long-term participation.
- Use crypto-centric indicators: TVL growth, user retention, governance engagement.
- Provide a data-driven recommendation (bullish, bearish, or neutral).

When providing your reasoning, be thorough and specific by:
1. Highlighting key on-chain adoption trends (active addresses, transaction count).
2. Discussing ecosystem health (developer contributions, GitHub activity, community size).
3. Analyzing total value locked or network revenue relative to market cap.
4. Explaining tokenomics, staking incentives, and supply dynamics.
5. Addressing governance participation and roadmap execution.
6. Using Cathie Wood’s optimistic, conviction-driven voice adapted for crypto.

For example, if bullish: “The network’s daily active addresses jumped from 30k to 120k in six months, TVL has grown 70% QoQ, and developer commits rose 40%, signaling rapid adoption in a $1T DeFi market…”  
If bearish: “Despite high TVL, NVT ratio is elevated, active address growth has plateaued, and developer activity is down 20%, indicating the token may be overvalued relative to its network utility…”"""
        ),
        (
            "human",
            """Based on the following analysis, create a Cathie Wood-style signal for crypto.

Analysis Data for {ticker}:
{analysis_data}

Initial Signal: {signal}

Return the signal as JSON:
{{
  "signal": "bullish/bearish/neutral",
  "confidence": float (0–100),
  "reasoning": "string"
}}"""
        ),
    ])

    prompt = template.invoke({
        "ticker": ticker,
        "analysis_data": json.dumps(analysis_data[ticker], indent=2),
        "signal": signal,
    })

    def create_default_cathie_wood_signal():
        return CathieWoodSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=CathieWoodSignal,
        agent_name="cathie_wood_agent",
        default_factory=create_default_cathie_wood_signal,
    )

# source: https://ark-invest.com