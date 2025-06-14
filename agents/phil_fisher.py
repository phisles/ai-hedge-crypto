from graph.state import AgentState, show_agent_reasoning
from tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
    get_company_news,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
import statistics


class PhilFisherSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def phil_fisher_agent(state: AgentState):
    """
    Analyzes crypto only using Phil Fisher's principles adapted for on-chain metrics.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    fisher_analysis: dict[str, dict] = {}

    for ticker in tickers:
        progress.update_status("phil_fisher_agent", ticker, "Fetching crypto metrics")
        metrics = get_financial_metrics(ticker, end_date)
        if not metrics:
            continue

        if ticker.upper().endswith(("/USD", "-USD")):
            latest = metrics[0]
            sentiment = analyze_crypto_sentiment_from_metrics(latest)

            # fetch on-chain & corporate data
            progress.update_status("phil_fisher_agent", ticker, "Fetching financial line items")
            financial_line_items = search_line_items(
                ticker=ticker,
                line_items=["revenue", "gross_profit", "price", "volume"],
                end_date=end_date
            )
            progress.update_status("phil_fisher_agent", ticker, "Fetching market cap")
            market_cap = get_market_cap(ticker, end_date)
            progress.update_status("phil_fisher_agent", ticker, "Fetching company news")
            company_news = get_company_news(ticker, end_date)

            # prepare everything for the LLM
            analysis_data = {
                "latest_metrics": latest,
                "market_cap": market_cap,
                "company_news": [n.dict() for n in company_news],
                "financial_line_items": [fi.dict() for fi in financial_line_items],
                "onchain_growth": {
                    "1y_change_pct": latest.get("price_change_pct_1y", 0.0),
                    "30d_change_pct": latest.get("price_change_pct_30d", 0.0),
                },
                "onchain_valuation": {
                    "volume_to_market_cap": latest.get("volume_to_market_cap", 0.0),
                    "developer_stars": latest.get("developer_stars", 0),
                },
                "community_sentiment": {
                    "sentiment_votes_up_pct": latest.get("sentiment_votes_up_pct", None)
                }
            }

            # ask the LLM to do the Phil-Fisher scoring & reasoning
            signal_obj = generate_fisher_output(
                ticker,
                analysis_data,
                state["metadata"]["model_name"],
                state["metadata"]["model_provider"],
            )

            fisher_analysis[ticker] = {
                "signal": signal_obj.signal,
                "confidence": signal_obj.confidence,
                "reasoning": signal_obj.reasoning,
            }
            if state["metadata"].get("show_reasoning"):
                show_agent_reasoning({ticker: signal_obj.dict()}, "Phil Fisher Agent")
            progress.update_status("phil_fisher_agent", ticker, "Done (LLM)")
            continue

    message = HumanMessage(content=json.dumps(fisher_analysis), name="phil_fisher_agent")
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(fisher_analysis, "Phil Fisher Agent")
    state["data"]["analyst_signals"]["phil_fisher_agent"] = fisher_analysis
    return {"messages": [message], "data": state["data"]}


def analyze_fisher_growth_quality(financial_line_items: list) -> dict:
    """
    Evaluate growth & quality:
      - Consistent Revenue Growth
      - Consistent EPS Growth
      - R&D as a % of Revenue (if relevant, indicative of future-oriented spending)
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {
            "score": 0,
            "details": "Insufficient financial data for growth/quality analysis",
        }

    details = []
    raw_score = 0  # up to 9 raw points => scale to 0–10

    # 1. Revenue Growth (YoY)
    revenues = [fi.revenue for fi in financial_line_items if fi.revenue is not None]
    if len(revenues) >= 2:
        # We'll look at the earliest vs. latest to gauge multi-year growth if possible
        latest_rev = revenues[0]
        oldest_rev = revenues[-1]
        if oldest_rev > 0:
            rev_growth = (latest_rev - oldest_rev) / abs(oldest_rev)
            if rev_growth > 0.80:
                raw_score += 3
                details.append(f"Very strong multi-period revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.40:
                raw_score += 2
                details.append(f"Moderate multi-period revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.10:
                raw_score += 1
                details.append(f"Slight multi-period revenue growth: {rev_growth:.1%}")
            else:
                details.append(f"Minimal or negative multi-period revenue growth: {rev_growth:.1%}")
        else:
            details.append("Oldest revenue is zero/negative; cannot compute growth.")
    else:
        details.append("Not enough revenue data points for growth calculation.")

    # 2. EPS Growth (YoY)
    eps_values = [fi.earnings_per_share for fi in financial_line_items if fi.earnings_per_share is not None]
    if len(eps_values) >= 2:
        latest_eps = eps_values[0]
        oldest_eps = eps_values[-1]
        if abs(oldest_eps) > 1e-9:
            eps_growth = (latest_eps - oldest_eps) / abs(oldest_eps)
            if eps_growth > 0.80:
                raw_score += 3
                details.append(f"Very strong multi-period EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.40:
                raw_score += 2
                details.append(f"Moderate multi-period EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.10:
                raw_score += 1
                details.append(f"Slight multi-period EPS growth: {eps_growth:.1%}")
            else:
                details.append(f"Minimal or negative multi-period EPS growth: {eps_growth:.1%}")
        else:
            details.append("Oldest EPS near zero; skipping EPS growth calculation.")
    else:
        details.append("Not enough EPS data points for growth calculation.")

    # 3. R&D as % of Revenue (if we have R&D data)
    rnd_values = [fi.research_and_development for fi in financial_line_items if fi.research_and_development is not None]
    if rnd_values and revenues and len(rnd_values) == len(revenues):
        # We'll just look at the most recent for a simple measure
        recent_rnd = rnd_values[0]
        recent_rev = revenues[0] if revenues[0] else 1e-9
        rnd_ratio = recent_rnd / recent_rev
        # Generally, Fisher admired companies that invest aggressively in R&D,
        # but it must be appropriate. We'll assume "3%-15%" is healthy, just as an example.
        if 0.03 <= rnd_ratio <= 0.15:
            raw_score += 3
            details.append(f"R&D ratio {rnd_ratio:.1%} indicates significant investment in future growth")
        elif rnd_ratio > 0.15:
            raw_score += 2
            details.append(f"R&D ratio {rnd_ratio:.1%} is very high (could be good if well-managed)")
        elif rnd_ratio > 0.0:
            raw_score += 1
            details.append(f"R&D ratio {rnd_ratio:.1%} is somewhat low but still positive")
        else:
            details.append("No meaningful R&D expense ratio")
    else:
        details.append("Insufficient R&D data to evaluate")

    # scale raw_score (max 9) to 0–10
    final_score = min(10, (raw_score / 9) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_margins_stability(financial_line_items: list) -> dict:
    """
    Looks at margin consistency (gross/operating margin) and general stability over time.
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {
            "score": 0,
            "details": "Insufficient data for margin stability analysis",
        }

    details = []
    raw_score = 0  # up to 6 => scale to 0-10

    # 1. Operating Margin Consistency
    op_margins = [fi.operating_margin for fi in financial_line_items if fi.operating_margin is not None]
    if len(op_margins) >= 2:
        # Check if margins are stable or improving (comparing oldest to newest)
        oldest_op_margin = op_margins[-1]
        newest_op_margin = op_margins[0]
        if newest_op_margin >= oldest_op_margin > 0:
            raw_score += 2
            details.append(f"Operating margin stable or improving ({oldest_op_margin:.1%} -> {newest_op_margin:.1%})")
        elif newest_op_margin > 0:
            raw_score += 1
            details.append(f"Operating margin positive but slightly declined")
        else:
            details.append(f"Operating margin may be negative or uncertain")
    else:
        details.append("Not enough operating margin data points")

    # 2. Gross Margin Level
    gm_values = [fi.gross_margin for fi in financial_line_items if fi.gross_margin is not None]
    if gm_values:
        # We'll just take the most recent
        recent_gm = gm_values[0]
        if recent_gm > 0.5:
            raw_score += 2
            details.append(f"Strong gross margin: {recent_gm:.1%}")
        elif recent_gm > 0.3:
            raw_score += 1
            details.append(f"Moderate gross margin: {recent_gm:.1%}")
        else:
            details.append(f"Low gross margin: {recent_gm:.1%}")
    else:
        details.append("No gross margin data available")

    # 3. Multi-year Margin Stability
    #   e.g. if we have at least 3 data points, see if standard deviation is low.
    if len(op_margins) >= 3:
        stdev = statistics.pstdev(op_margins)
        if stdev < 0.02:
            raw_score += 2
            details.append("Operating margin extremely stable over multiple years")
        elif stdev < 0.05:
            raw_score += 1
            details.append("Operating margin reasonably stable")
        else:
            details.append("Operating margin volatility is high")
    else:
        details.append("Not enough margin data points for volatility check")

    # scale raw_score (max 6) to 0-10
    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_management_efficiency_leverage(financial_line_items: list) -> dict:
    """
    Evaluate management efficiency & leverage:
      - Return on Equity (ROE)
      - Debt-to-Equity ratio
      - Possibly check if free cash flow is consistently positive
    """
    if not financial_line_items:
        return {
            "score": 0,
            "details": "No financial data for management efficiency analysis",
        }

    details = []
    raw_score = 0  # up to 6 => scale to 0–10

    # 1. Return on Equity (ROE)
    ni_values = [fi.net_income for fi in financial_line_items if fi.net_income is not None]
    eq_values = [fi.shareholders_equity for fi in financial_line_items if fi.shareholders_equity is not None]
    if ni_values and eq_values and len(ni_values) == len(eq_values):
        recent_ni = ni_values[0]
        recent_eq = eq_values[0] if eq_values[0] else 1e-9
        if recent_ni > 0:
            roe = recent_ni / recent_eq
            if roe > 0.2:
                raw_score += 3
                details.append(f"High ROE: {roe:.1%}")
            elif roe > 0.1:
                raw_score += 2
                details.append(f"Moderate ROE: {roe:.1%}")
            elif roe > 0:
                raw_score += 1
                details.append(f"Positive but low ROE: {roe:.1%}")
            else:
                details.append(f"ROE is near zero or negative: {roe:.1%}")
        else:
            details.append("Recent net income is zero or negative, hurting ROE")
    else:
        details.append("Insufficient data for ROE calculation")

    # 2. Debt-to-Equity
    debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
    if debt_values and eq_values and len(debt_values) == len(eq_values):
        recent_debt = debt_values[0]
        recent_equity = eq_values[0] if eq_values[0] else 1e-9
        dte = recent_debt / recent_equity
        if dte < 0.3:
            raw_score += 2
            details.append(f"Low debt-to-equity: {dte:.2f}")
        elif dte < 1.0:
            raw_score += 1
            details.append(f"Manageable debt-to-equity: {dte:.2f}")
        else:
            details.append(f"High debt-to-equity: {dte:.2f}")
    else:
        details.append("Insufficient data for debt/equity analysis")

    # 3. FCF Consistency
    fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]
    if fcf_values and len(fcf_values) >= 2:
        # Check if FCF is positive in recent years
        positive_fcf_count = sum(1 for x in fcf_values if x and x > 0)
        # We'll be simplistic: if most are positive, reward
        ratio = positive_fcf_count / len(fcf_values)
        if ratio > 0.8:
            raw_score += 1
            details.append(f"Majority of periods have positive FCF ({positive_fcf_count}/{len(fcf_values)})")
        else:
            details.append(f"Free cash flow is inconsistent or often negative")
    else:
        details.append("Insufficient or no FCF data to check consistency")

    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_fisher_valuation(financial_line_items: list, market_cap: float | None) -> dict:
    """
    Phil Fisher is willing to pay for quality and growth, but still checks:
      - P/E
      - P/FCF
      - (Optionally) Enterprise Value metrics, but simpler approach is typical
    We will grant up to 2 points for each of two metrics => max 4 raw => scale to 0–10.
    """
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    details = []
    raw_score = 0

    # Gather needed data
    net_incomes = [fi.net_income for fi in financial_line_items if fi.net_income is not None]
    fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]

    # 1) P/E
    recent_net_income = net_incomes[0] if net_incomes else None
    if recent_net_income and recent_net_income > 0:
        pe = market_cap / recent_net_income
        pe_points = 0
        if pe < 20:
            pe_points = 2
            details.append(f"Reasonably attractive P/E: {pe:.2f}")
        elif pe < 30:
            pe_points = 1
            details.append(f"Somewhat high but possibly justifiable P/E: {pe:.2f}")
        else:
            details.append(f"Very high P/E: {pe:.2f}")
        raw_score += pe_points
    else:
        details.append("No positive net income for P/E calculation")

    # 2) P/FCF
    recent_fcf = fcf_values[0] if fcf_values else None
    if recent_fcf and recent_fcf > 0:
        pfcf = market_cap / recent_fcf
        pfcf_points = 0
        if pfcf < 20:
            pfcf_points = 2
            details.append(f"Reasonable P/FCF: {pfcf:.2f}")
        elif pfcf < 30:
            pfcf_points = 1
            details.append(f"Somewhat high P/FCF: {pfcf:.2f}")
        else:
            details.append(f"Excessively high P/FCF: {pfcf:.2f}")
        raw_score += pfcf_points
    else:
        details.append("No positive free cash flow for P/FCF calculation")

    # scale raw_score (max 4) to 0–10
    final_score = min(10, (raw_score / 4) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_insider_activity(insider_trades: list) -> dict:
    """
    Simple insider-trade analysis:
      - If there's heavy insider buying, we nudge the score up.
      - If there's mostly selling, we reduce it.
      - Otherwise, neutral.
    """
    # Default is neutral (5/10).
    score = 5
    details = []

    if not insider_trades:
        details.append("No insider trades data; defaulting to neutral")
        return {"score": score, "details": "; ".join(details)}

    buys, sells = 0, 0
    for trade in insider_trades:
        if trade.transaction_shares is not None:
            if trade.transaction_shares > 0:
                buys += 1
            elif trade.transaction_shares < 0:
                sells += 1

    total = buys + sells
    if total == 0:
        details.append("No buy/sell transactions found; neutral")
        return {"score": score, "details": "; ".join(details)}

    buy_ratio = buys / total
    if buy_ratio > 0.7:
        score = 8
        details.append(f"Heavy insider buying: {buys} buys vs. {sells} sells")
    elif buy_ratio > 0.4:
        score = 6
        details.append(f"Moderate insider buying: {buys} buys vs. {sells} sells")
    else:
        score = 4
        details.append(f"Mostly insider selling: {buys} buys vs. {sells} sells")

    return {"score": score, "details": "; ".join(details)}


def analyze_crypto_sentiment_from_metrics(metrics: dict) -> dict:
    """
    Uses the built-in sentiment_votes_up_pct from your crypto metrics.
    Maps 0–100% upvotes into a 0–10 score.
    """
    up_pct = metrics.get("sentiment_votes_up_pct")
    if up_pct is None:
        return {"score": 5, "details": "No sentiment data"}

    score = round(up_pct / 10, 1)
    return {
        "score": score,
        "details": f"{up_pct:.1f}% positive community votes"
    }




def generate_fisher_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> PhilFisherSignal:
    """
    Generates a final JSON signal in Phil Fisher's style, adapted for crypto.
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Phil Fisher–style AI agent specializing in cryptocurrency assets.

1. Emphasize long-term growth potential via 1-year price change.
2. Assess momentum stability with 30-day price trends.
3. Evaluate developer activity as a proxy for management quality and R&D.
4. Use volume-to-market-cap as a simple valuation signal.
5. Include community sentiment (positive vote percentage) as a check.
6. Be willing to pay a premium for quality but remain mindful of valuation.
7. Provide detailed reasoning on each metric.
8. Conclude with a clear stance (bullish, bearish, or neutral) and confidence.

Return only valid JSON:
{{
  "signal": "bullish"|"bearish"|"neutral",
  "confidence": 0–100,
  "reasoning": "string"
}}"""
            ),
            (
                "human",
                """Based on the following crypto analysis for {ticker}:
{analysis_data}

Return your Phil Fisher–style crypto signal exactly as JSON."""
            ),
        ]
    )

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def default_signal():
        return PhilFisherSignal(signal="neutral", confidence=0.0, reasoning="Error; defaulting to neutral")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=PhilFisherSignal,
        agent_name="phil_fisher_agent",
        default_factory=default_signal,
    )