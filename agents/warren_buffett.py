from graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from tools.api import get_financial_metrics, get_market_cap
from utils.llm import call_llm
from utils.progress import progress


class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


##### Crypto-Adapted Buffett Agent #####
def warren_buffett_agent(state: AgentState):
    """
    Applies Warren Buffett's principles to crypto assets using on-chain parallels:
      - Circle of Competence: use metrics we understand (NVT, addr growth)
      - Margin of Safety: NVT-based discount to market value
      - Economic Moat: active-address consistency & developer activity
      - Management Quality: inflation rate & staking yield
      - Long-term Horizon: favor network strength over short-term price swings
      - Sell only if on-chain fundamentals deteriorate or valuation far exceeds intrinsic value
    """
    data = state["data"]
    end_date = data.get("end_date")
    tickers = data.get("tickers", [])

    analysis_data = {}
    buffett_analysis = {}

    for ticker in tickers:
        progress.update_status("warren_buffett_agent", ticker, "Fetching metrics")
        metrics_list = get_financial_metrics(ticker, end_date)
        if not metrics_list:
            progress.update_status("warren_buffett_agent", ticker, "No metrics found")
            continue
        # latest and historical
        metrics = metrics_list[0]

        progress.update_status("warren_buffett_agent", ticker, "Fetching market cap")
        market_cap = get_market_cap(ticker=ticker, end_date=end_date) or 0.0

        # Fundamental analysis
        progress.update_status("warren_buffett_agent", ticker, "Analyzing fundamentals")
        fundamental = analyze_fundamentals(metrics)

        # Consistency of network growth
        progress.update_status("warren_buffett_agent", ticker, "Analyzing consistency")
        consistency = analyze_consistency(metrics_list)

        # Moat: dev activity stability
        progress.update_status("warren_buffett_agent", ticker, "Analyzing moat")
        moat = analyze_moat(metrics_list)

        # Management: tokenomics
        progress.update_status("warren_buffett_agent", ticker, "Analyzing management quality")
        management = analyze_management_quality(metrics)

        # Margin of Safety via NVT
        total_volume = metrics.get("total_volume", 0.0)
        nvt_ratio = None
        if total_volume > 0:
            nvt_ratio = market_cap / total_volume
        margin_of_safety = None
        if nvt_ratio is not None and market_cap > 0:
            margin_of_safety = (40 - nvt_ratio) / nvt_ratio

        # Composite scoring
        total_score = fundamental["score"] + consistency["score"] + management["score"]
        max_score = fundamental.get("max_score", 10) + consistency.get("max_score", 2) + management.get("max_score", 2)

        # Determine signal
        if total_score >= 0.7 * max_score and margin_of_safety is not None and margin_of_safety >= 0.3:
            signal = "bullish"
        elif total_score <= 0.3 * max_score or (margin_of_safety is not None and margin_of_safety < -0.3):
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "fundamental_analysis": fundamental,
            "consistency_analysis": consistency,
            "moat_analysis": moat,
            "management_analysis": management,
            "nvt_ratio": nvt_ratio,
            "margin_of_safety": margin_of_safety,
            "score": total_score,
            "max_score": max_score,
            "market_cap": market_cap,
        }

        progress.update_status("warren_buffett_agent", ticker, "Generating LLM signal")
        buffett_output = generate_buffett_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            signal=signal,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        buffett_analysis[ticker] = {
            "signal": buffett_output.signal,
            "confidence": buffett_output.confidence,
            "reasoning": buffett_output.reasoning,
        }

        progress.update_status("warren_buffett_agent", ticker, "Done")

    message = HumanMessage(content=json.dumps(buffett_analysis), name="warren_buffett_agent")
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(buffett_analysis, "Crypto Buffett Agent")

    state["data"]["analyst_signals"]["warren_buffett_agent"] = buffett_analysis
    return {"messages": [message], "data": state["data"]}


def analyze_fundamentals(metrics: dict) -> dict:
    """Score NVT, address growth, and developer activity as crypto fundamentals."""
    score = 0
    details = []
    # NVT component
    cap = metrics.get("market_cap", 0.0)
    vol = metrics.get("total_volume", 0.0)
    if vol > 0 and cap > 0:
        nvt = cap / vol
        if nvt < 20:
            score += 3
            details.append(f"Low NVT ratio {nvt:.1f}")
        elif nvt > 80:
            score -= 2
            details.append(f"High NVT ratio {nvt:.1f}")
        else:
            details.append(f"Neutral NVT ratio {nvt:.1f}")
    else:
        details.append("Insufficient NVT data")

    # Active address growth
    ag = metrics.get("active_addresses_24h")
    if ag and ag > 0:
        score += 2
        details.append(f"Address growth {ag}")
    else:
        details.append("No address growth data")


    return {"score": score, "max_score": 10, "details": "; ".join(details), "metrics": metrics}


def analyze_consistency(metrics_list: list) -> dict:
    """Check consistency of active address growth over multiple periods."""
    if len(metrics_list) < 3:
        return {"score": 0, "max_score": 2, "details": "Insufficient data for consistency"}
    adds = [m.get("active_addresses_24h", 0) for m in metrics_list]
    if all(adds[i] >= adds[i+1] for i in range(len(adds)-1)):
        return {"score": 2, "max_score": 2, "details": "Consistent address growth"}
    return {"score": 0, "max_score": 2, "details": "Inconsistent address trend"}


def analyze_moat(metrics_list: list) -> dict:
    """Evaluate network moat via developer activity stability."""
    if len(metrics_list) < 2:
        return {"score": 0, "max_score": 3, "details": "Insufficient data for moat"}
    scores = [1 if (m.get("developer_activity") or 0) >= 50 else 0 for m in metrics_list]
    moat_score = sum(scores)
    details = f"Dev activity consistency {moat_score}/{len(metrics_list)}"
    return {"score": moat_score, "max_score": len(metrics_list), "details": details}


def analyze_management_quality(metrics: dict) -> dict:
    """Assess tokenomics: inflation rate and staking yield."""
    score = 0
    details = []
    inf = metrics.get("inflation_rate")
    if inf is not None:
        if inf < 0.05:
            score += 1
            details.append(f"Low inflation {inf:.1%}")
        else:
            details.append(f"High inflation {inf:.1%}")
    sy = metrics.get("staking_yield")
    if sy is not None:
        if sy > 0.05:
            score += 1
            details.append(f"High staking yield {sy:.1%}")
        else:
            details.append(f"Low staking yield {sy:.1%}")
    return {"score": score, "max_score": 2, "details": "; ".join(details)}


def generate_buffett_output(
    ticker: str,
    analysis_data: dict,
    signal: str,
    model_name: str,
    model_provider: str,
) -> WarrenBuffettSignal:
    """Get Crypto Buffett signal from LLM."""
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a crypto-adapted Warren Buffett AI, making decisions on digital assets using these principles:
- Circle of Competence: Only invest in protocols you understand (tokenomics, use-cases, tech stack)
- Margin of Safety: Require a significant discount to intrinsic or network-value metrics (e.g. NVT)
- Economic Moat: Seek durable network effects and high active-address growth
- Quality Management: Value clear tokenomics and low inflation
- Financial Strength: Favor tokens with low inflation rates and attractive staking yields
- Long-term Horizon: Hold assets for network maturation, not just price swings
- Sell only if on-chain fundamentals deteriorate, tokenomics break down, or valuation far exceeds intrinsic value

When providing your reasoning, be thorough and specific by:
1. Explaining which on-chain and tokenomics factors mattered most (e.g., NVT ratio, address growth)
2. Showing how the asset aligns with or violates these Buffett-inspired principles
3. Providing quantitative evidence from analysis_data (e.g., NVT thresholds, inflation rate)
4. Concluding with a Buffett-style conviction statement in a clear, conversational tone
"""
        ),
        (
            "human",
            """Analysis for {ticker}:
{analysis_data}

Return JSON exactly:
{{"signal":"bullish/bearish/neutral","confidence":float,"reasoning":"string"}}
"""
        ),
    ])

    prompt = template.invoke({
        "ticker": ticker,
        "analysis_data": analysis_data
    })

    def default():
        return WarrenBuffettSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis; defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=WarrenBuffettSignal,
        agent_name="warren_buffett_agent",
        default_factory=default,
    )
