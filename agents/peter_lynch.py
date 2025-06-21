from graph.state import AgentState, show_agent_reasoning
from tools.api import (
    get_financial_metrics,
    get_market_cap,
    get_company_news,
    get_prices,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
from langchain_core.prompts import ChatPromptTemplate
import json
from utils.llm import call_llm
from data.models import PeterLynchSignal


class PeterLynchSignal(BaseModel):
    """
    Container for the Peter Lynch–style output signal.
    """
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def peter_lynch_agent(state: AgentState):
    """
    Analyzes crypto and stocks using Peter Lynch's style, with a full crypto branch.
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    lynch_analysis: dict[str, dict] = {}

    for ticker in tickers:
        progress.update_status("peter_lynch_agent", ticker, "Fetching metrics")
        metrics = get_financial_metrics(ticker, end_date)
        if not metrics:
            continue

        # —— CRYPTO BRANCH —— 
        if ticker.upper().endswith(("/USD", "-USD")):
            # -- Base metrics
            latest = get_financial_metrics(ticker, end_date)[0]
            price_30d = latest.get("price_change_pct_30d", 0.0)
            sentiment_data = analyze_crypto_sentiment_from_metrics(latest)
            vol_mc = latest.get("volume_to_market_cap", 0.0)

            # -- Optional enhancements
            try:
                market_cap = get_market_cap(ticker, end_date)
            except Exception:
                market_cap = None

            try:
                prices = get_prices(ticker, start_date=start_date, end_date=end_date)
            except Exception:
                prices = []

            try:
                company_news = get_company_news(ticker, end_date, start_date=start_date, limit=50)
                from collections import Counter
                sentiment_counts = Counter(n.sentiment for n in company_news if n.sentiment)
                total_articles = sum(sentiment_counts.values())
                news_sentiment = {
                    "positive": sentiment_counts.get("positive", 0),
                    "neutral": sentiment_counts.get("neutral", 0),
                    "negative": sentiment_counts.get("negative", 0),
                    "summary": f"{sentiment_counts.get('positive', 0)}↑ / "
                            f"{sentiment_counts.get('neutral', 0)}→ / "
                            f"{sentiment_counts.get('negative', 0)}↓",
                    "score": (
                        (sentiment_counts.get("positive", 0) - sentiment_counts.get("negative", 0)) / total_articles
                        if total_articles > 0 else 0
                    )
                }
            except Exception:
                news_sentiment = {"score": 0, "summary": "No news available"}

            # -- Scoring (same base logic)
            score = 0
            if price_30d is not None:
                if price_30d > 0.20:
                    score += 2
                elif price_30d > 0.05:
                    score += 1

            if sentiment_data.get("score") is not None and sentiment_data["score"] >= 7:
                score += 1

            if vol_mc is not None and vol_mc > 0.03:
                score += 1


            if score >= 4:
                signal = "bullish"
            elif score == 0:
                signal = "bearish"
            else:
                signal = "neutral"

            confidence = round(min(score / 5, 1.0) * 100)

            lynch_output = generate_lynch_output(
                ticker=ticker,
                analysis_data={
                    "30d_price_change_pct": price_30d,
                    "sentiment": sentiment_data,
                    "volume_to_market_cap": vol_mc,
                    "developer_stars": dev_stars,
                    "market_cap": market_cap,
                    "price_history_count": len(prices),
                    "news_sentiment": news_sentiment,
                },
                model_name=state["metadata"]["model_name"],
                model_provider=state["metadata"]["model_provider"],
            )

            lynch_analysis[ticker] = lynch_output.dict()
            progress.update_status("peter_lynch_agent", ticker, "Done (crypto)")
            continue
        # —— END CRYPTO BRANCH —— 
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(lynch_analysis, "Peter Lynch Agent")

    # store into state
    state["data"]["analyst_signals"]["peter_lynch_agent"] = lynch_analysis

    # <— FIXED HERE:
    return {
        "messages": [
            HumanMessage(
                content=json.dumps(lynch_analysis),
                name="peter_lynch_agent"
            )
        ],
        "data": state["data"]
    }











def analyze_crypto_sentiment_from_metrics(metrics: dict) -> dict:
    """
    Uses the built-in sentiment_votes_up_pct from your crypto metrics.
    Maps 0–100% upvotes into a 0–10 score.
    """
    up_pct = metrics.get("sentiment_votes_up_pct")
    if up_pct is None:
        return {"score": 5, "details": "No sentiment data"}

    # 0% → 0, 50% → 5, 100% → 10
    score = round(up_pct / 10, 1)
    return {
        "score": score,
        "details": f"{up_pct:.1f}% positive community votes"
    }







def generate_lynch_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> PeterLynchSignal:
    """
    Generates a final JSON signal in Peter Lynch's style, adapted for crypto.
    """
    # 1) Build the system+human prompt template
    template = ChatPromptTemplate.from_messages([
        ("system", """You are a Peter Lynch–style AI agent specializing in cryptocurrency assets.

    1. Invest in What You Know: favor protocols with understandable use cases and real users.
    2. Growth at a Reasonable Price: prioritize on-chain growth (e.g. users, devs, TVL).
    3. Seek 'Ten-Baggers': identify high-potential assets with small market caps and strong momentum.
    4. Avoid complexity: penalize excessive inflation, broken tokenomics, or hype-based pricing.
    5. Use community sentiment and liquidity as supporting evidence.
    6. Be direct and use clear language with reasoning grounded in data.
    7. Conclude with a clear stance: bullish, bearish, or neutral.

    Respond only with JSON:
    {{
    "signal": "bullish" | "bearish" | "neutral",
    "confidence": 0–100,
    "reasoning": "string"
    }}"""),
        ("human", """Based on the following analysis for {ticker}:
    {analysis_data}

    Return your Peter Lynch–style signal exactly as JSON."""),
    ])

    # 2) Fill in the template
    prompt = template.invoke({
        "ticker": ticker,
        "analysis_data": json.dumps(analysis_data, indent=2)
    })

    # 3) Default in case the LLM call fails
    def default_signal():
        return PeterLynchSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis; defaulting to neutral"
        )

    # 4) Call the LLM and parse into our Pydantic model
    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=PeterLynchSignal,
        agent_name="peter_lynch_agent",
        default_factory=default_signal,
    )
