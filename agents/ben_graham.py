from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
import math


class BenGrahamSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def ben_graham_agent(state: AgentState):
    """
    Analyzes stocks using Benjamin Graham's classic value-investing principles:
    1. Earnings stability over multiple years.
    2. Solid financial strength (low debt, adequate liquidity).
    3. Discount to intrinsic value (e.g. Graham Number or net-net).
    4. Adequate margin of safety.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    graham_analysis = {}

    for ticker in tickers:
        progress.update_status("ben_graham_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date)

        progress.update_status("ben_graham_agent", ticker, "Gathering financial line items")
        financial_line_items = []
        progress.update_status("ben_graham_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

        # Perform sub-analyses
        progress.update_status("ben_graham_agent", ticker, "Analyzing price stability")
        earnings_analysis = analyze_price_stability_crypto(metrics)

        progress.update_status("ben_graham_agent", ticker, "Analyzing liquidity strength")
        strength_analysis = analyze_liquidity_strength_crypto(metrics)

        progress.update_status("ben_graham_agent", ticker, "Analyzing crypto valuation")
        valuation_analysis = analyze_valuation_crypto(metrics, financial_line_items, market_cap)

        # Aggregate scoring
        total_score = earnings_analysis["score"] + strength_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15  # total possible from the three analysis functions

        # Map total_score to signal
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {"signal": signal, "score": total_score, "max_score": max_possible_score, "earnings_analysis": earnings_analysis, "strength_analysis": strength_analysis, "valuation_analysis": valuation_analysis}

        progress.update_status("ben_graham_agent", ticker, "Generating Ben Graham analysis")
        graham_output = generate_graham_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        graham_analysis[ticker] = {"signal": graham_output.signal, "confidence": graham_output.confidence, "reasoning": graham_output.reasoning}

        progress.update_status("ben_graham_agent", ticker, "Done")

    # Wrap results in a single message for the chain
    message = HumanMessage(content=json.dumps(graham_analysis), name="ben_graham_agent")

    # Optionally display reasoning
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(graham_analysis, "Ben Graham Agent")

    # Store signals in the overall state
    state["data"]["analyst_signals"]["ben_graham_agent"] = graham_analysis

    return {"messages": [message], "data": state["data"]}


def analyze_price_stability_crypto(metrics: list, financial_line_items: list = None) -> dict:
    """
    For crypto, proxy for 'earnings stability' by checking multi-period price stability.
    If price has grown or remained stable over time, treat as bullish.
    """
    score = 0
    details = []

    if not metrics or not isinstance(metrics, list) or len(metrics) == 0:
        return {"score": score, "details": "No metrics data available"}

    m = metrics[0]
    pct_1y = m.get("price_change_pct_1y")
    pct_200d = m.get("price_change_pct_200d")
    pct_60d = m.get("price_change_pct_60d")

    if pct_1y and pct_1y > 0:
        score += 2
        details.append(f"1Y price change positive: {pct_1y:.2f}%")
    elif pct_1y and pct_1y > -10:
        score += 1
        details.append(f"1Y drawdown mild: {pct_1y:.2f}%")
    else:
        details.append(f"1Y price drop: {pct_1y:.2f}%")

    if pct_200d and pct_200d > 0:
        score += 1
        details.append(f"200d price momentum also positive: {pct_200d:.2f}%")

    return {"score": score, "details": "; ".join(details)}


def analyze_liquidity_strength_crypto(metrics: list, financial_line_items: list = None) -> dict:
    """
    For crypto, use volume and market cap to proxy liquidity and demand.
    """
    score = 0
    details = []

    if not metrics or not isinstance(metrics, list):
        return {"score": score, "details": "No metrics available"}

    m = metrics[0]
    volume = m.get("volume_24h")
    market_cap = m.get("market_cap")
    ratio = m.get("volume_to_market_cap")

    if ratio:
        if ratio > 0.1:
            score += 2
            details.append(f"High volume/market cap ratio: {ratio:.2f} (strong liquidity)")
        elif ratio > 0.05:
            score += 1
            details.append(f"Moderate liquidity: ratio={ratio:.2f}")
        else:
            details.append(f"Low liquidity: ratio={ratio:.2f}")
    elif volume and market_cap:
        v_m_ratio = volume / market_cap
        details.append(f"Estimated volume/market cap ratio: {v_m_ratio:.2f}")

    return {"score": score, "details": "; ".join(details)}


def analyze_valuation_crypto(metrics: list, financial_line_items: list, market_cap: float) -> dict:
    """
    Use crypto-native valuation: price relative to ATH, P/S ratio, and trend.
    """
    score = 0
    details = []

    if not metrics or not isinstance(metrics, list):
        return {"score": score, "details": "No metrics available"}

    m = metrics[0]
    price = m.get("current_price")
    ath = m.get("ath")
    ps_ratio = m.get("price_to_sales_ratio")

    if price and ath:
        distance_from_ath = (ath - price) / ath
        details.append(f"Price is {(distance_from_ath * 100):.2f}% below ATH")
        if distance_from_ath >= 0.5:
            score += 2
            details.append("Price is significantly below ATH (>= 50%)")
        elif distance_from_ath >= 0.25:
            score += 1
            details.append("Price is moderately below ATH (>= 25%)")

    if ps_ratio:
        if ps_ratio < 10:
            score += 2
            details.append(f"Low P/S ratio: {ps_ratio:.2f} (undervalued)")
        elif ps_ratio < 20:
            score += 1
            details.append(f"Moderate P/S ratio: {ps_ratio:.2f}")
        else:
            details.append(f"High P/S ratio: {ps_ratio:.2f}")

    return {"score": score, "details": "; ".join(details)}


def generate_graham_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> BenGrahamSignal:
    """
    Generates an investment decision in the style of Benjamin Graham:
    - Value emphasis, margin of safety, net-nets, conservative balance sheet, stable earnings.
    - Return the result in a JSON structure: { signal, confidence, reasoning }.
    """

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Benjamin Graham-style crypto analyst. Your job is to assess the investment soundness of cryptocurrency assets using conservative value principles adapted to digital assets.

            1. Require a margin of safety: look for assets trading well below their all-time highs or with low price-to-sales ratios.
            2. Emphasize financial strength and liquidity: prefer projects with high trading volume relative to market cap.
            3. Prioritize price stability over speculation: assets with consistent price performance over time are preferred.
            4. Avoid hype-based, highly volatile, or speculative assets lacking real usage or demand.
            5. Focus on hard data — prefer undervalued, well-traded, and established projects with defensible fundamentals.

            In your reasoning:
            - Explain key valuation metrics like P/S ratio and drawdown from ATH.
            - Highlight trading liquidity (volume-to-market-cap ratio) and its implications.
            - Comment on multi-period price stability and whether the asset shows signs of long-term viability.
            - Use specific numbers and thresholds (e.g., "Price is 45% below ATH" or "P/S ratio of 8 is below the conservative 10 threshold").
            - Avoid optimism based on future potential — focus only on current and historical financial facts.
            - Use a calm, cautious, and analytical tone in the voice of Benjamin Graham.

            Return your final recommendation (signal: bullish, neutral, or bearish) with a 0-100 confidence score and well-supported reasoning.
            """
        ),
        (
            "human",
            """Based on the following analysis, create a Graham-style investment signal:

            Analysis Data for {ticker}:
            {analysis_data}

            Return JSON exactly in this format:
            {{
              "signal": "bullish" or "bearish" or "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def create_default_ben_graham_signal():
        return BenGrahamSignal(signal="neutral", confidence=0.0, reasoning="Error in generating analysis; defaulting to neutral.")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=BenGrahamSignal,
        agent_name="ben_graham_agent",
        default_factory=create_default_ben_graham_signal,
    )