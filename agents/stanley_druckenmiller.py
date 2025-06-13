from graph.state import AgentState, show_agent_reasoning
from tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
    get_insider_trades,
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
import statistics


class StanleyDruckenmillerSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def stanley_druckenmiller_agent(state: AgentState):
    """
    Crypto‐focused Stanley Druckenmiller agent:
      - Emphasizes price momentum & volatility for asymmetric risk‐reward
      - Analyzes whale (large‐holder) transactions via get_insider_trades
      - Assesses crypto news sentiment
      - Incorporates basic token metrics (market cap, volume)
    """
    data = state["data"]
    start_date, end_date = data["start_date"], data["end_date"]
    tickers = data["tickers"]
    druck_analysis = {}

    for ticker in tickers:
        progress.update_status("stanley_druckenmiller_agent", ticker, "Fetching on‐chain metrics")
        metrics_list = get_financial_metrics(ticker)
        metrics = metrics_list[0] if isinstance(metrics_list, list) and metrics_list else {}
        market_cap = metrics.get("market_cap")
        total_volume = metrics.get("total_volume")

        progress.update_status("stanley_druckenmiller_agent", ticker, "Fetching whale trades")
        whale_trades = get_insider_trades(ticker, end_date, start_date=start_date, limit=50)

        progress.update_status("stanley_druckenmiller_agent", ticker, "Fetching crypto news")
        news = get_company_news(ticker, end_date, start_date=start_date, limit=50)

        progress.update_status("stanley_druckenmiller_agent", ticker, "Fetching price history")
        prices = get_prices(ticker, start_date=start_date, end_date=end_date)

        mom = analyze_momentum(prices)
        vol = analyze_risk_reward(prices)
        sent = analyze_sentiment(news)
        whale = analyze_insider_activity(whale_trades)

        # combine scores: momentum 30%, volatility 30%, sentiment 20%, whale 20%
        score = mom["score"] * 0.3 + vol["score"] * 0.3 + sent["score"] * 0.2 + whale["score"] * 0.2
        if score >= 7.5:
            signal = "bullish"
        elif score <= 4.5:
            signal = "bearish"
        else:
            signal = "neutral"

        reasoning = (
            f"Price momentum: {mom['details']}; "
            f"Volatility: {vol['details']}; "
            f"Sentiment: {sent['details']}; "
            f"Whale activity: {whale['details']}"
        )

        druck_analysis[ticker] = {
            "signal": signal,
            "confidence": round(score * 10, 1),
            "reasoning": reasoning,
        }
        progress.update_status("stanley_druckenmiller_agent", ticker, "Done")

    message = HumanMessage(content=json.dumps(druck_analysis), name="stanley_druckenmiller_agent")
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(druck_analysis, "Stanley Druckenmiller Agent")

    state["data"]["analyst_signals"]["stanley_druckenmiller_agent"] = druck_analysis
    return {"messages": [message], "data": state["data"]}


def analyze_momentum(prices: list) -> dict:
    if not prices or len(prices) < 2:
        return {"score": 5, "details": "Insufficient price data"}
    sorted_p = sorted(prices, key=lambda p: p.time)
    start, end = sorted_p[0].close, sorted_p[-1].close
    if start <= 0:
        return {"score": 5, "details": "Invalid start price"}
    pct = (end - start) / start
    if pct > 0.5:
        score, details = 10, f"Up {pct:.1%} over period"
    elif pct > 0.2:
        score, details = 7, f"Up {pct:.1%} over period"
    elif pct > 0:
        score, details = 5, f"Up {pct:.1%} over period"
    else:
        score, details = 2, f"Down {pct:.1%} over period"
    return {"score": score, "details": details}


def analyze_growth_and_momentum(financial_line_items: list, prices: list) -> dict:
    """
    Evaluate:
      - Revenue Growth (YoY)
      - EPS Growth (YoY)
      - Price Momentum
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {"score": 0, "details": "Insufficient financial data for growth analysis"}

    details = []
    raw_score = 0  # We'll sum up a maximum of 9 raw points, then scale to 0–10

    #
    # 1. Revenue Growth
    #
    revenues = [fi.revenue for fi in financial_line_items if fi.revenue is not None]
    if len(revenues) >= 2:
        latest_rev = revenues[0]
        older_rev = revenues[-1]
        if older_rev > 0:
            rev_growth = (latest_rev - older_rev) / abs(older_rev)
            if rev_growth > 0.30:
                raw_score += 3
                details.append(f"Strong revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.15:
                raw_score += 2
                details.append(f"Moderate revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.05:
                raw_score += 1
                details.append(f"Slight revenue growth: {rev_growth:.1%}")
            else:
                details.append(f"Minimal/negative revenue growth: {rev_growth:.1%}")
        else:
            details.append("Older revenue is zero/negative; can't compute revenue growth.")
    else:
        details.append("Not enough revenue data points for growth calculation.")

    #
    # 2. EPS Growth
    #
    eps_values = [fi.earnings_per_share for fi in financial_line_items if fi.earnings_per_share is not None]
    if len(eps_values) >= 2:
        latest_eps = eps_values[0]
        older_eps = eps_values[-1]
        # Avoid division by zero
        if abs(older_eps) > 1e-9:
            eps_growth = (latest_eps - older_eps) / abs(older_eps)
            if eps_growth > 0.30:
                raw_score += 3
                details.append(f"Strong EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.15:
                raw_score += 2
                details.append(f"Moderate EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.05:
                raw_score += 1
                details.append(f"Slight EPS growth: {eps_growth:.1%}")
            else:
                details.append(f"Minimal/negative EPS growth: {eps_growth:.1%}")
        else:
            details.append("Older EPS is near zero; skipping EPS growth calculation.")
    else:
        details.append("Not enough EPS data points for growth calculation.")

    #
    # 3. Price Momentum
    #
    # We'll give up to 3 points for strong momentum
    if prices and len(prices) > 30:
        sorted_prices = sorted(prices, key=lambda p: p.time)
        close_prices = [p.close for p in sorted_prices if p.close is not None]
        if len(close_prices) >= 2:
            start_price = close_prices[0]
            end_price = close_prices[-1]
            if start_price > 0:
                pct_change = (end_price - start_price) / start_price
                if pct_change > 0.50:
                    raw_score += 3
                    details.append(f"Very strong price momentum: {pct_change:.1%}")
                elif pct_change > 0.20:
                    raw_score += 2
                    details.append(f"Moderate price momentum: {pct_change:.1%}")
                elif pct_change > 0:
                    raw_score += 1
                    details.append(f"Slight positive momentum: {pct_change:.1%}")
                else:
                    details.append(f"Negative price momentum: {pct_change:.1%}")
            else:
                details.append("Invalid start price (<= 0); can't compute momentum.")
        else:
            details.append("Insufficient price data for momentum calculation.")
    else:
        details.append("Not enough recent price data for momentum analysis.")

    # We assigned up to 3 points each for:
    #   revenue growth, eps growth, momentum
    # => max raw_score = 9
    # Scale to 0–10
    final_score = min(10, (raw_score / 9) * 10)

    return {"score": final_score, "details": "; ".join(details)}


def analyze_insider_activity(insider_trades: list) -> dict:
    if not insider_trades:
        return {"score": 5, "details": "No whale trades data"}
    buys = sum(1 for t in insider_trades if getattr(t, 'transaction_shares', 0) > 0)
    sells = sum(1 for t in insider_trades if getattr(t, 'transaction_shares', 0) < 0)
    total = buys + sells
    if total == 0:
        return {"score": 5, "details": "No whale buy/sell detected"}
    ratio = buys / total
    if ratio > 0.7:
        score, details = 8, f"Heavy whale buying: {buys}/{total}"  
    elif ratio > 0.4:
        score, details = 6, f"Moderate whale buying: {buys}/{total}"  
    else:
        score, details = 4, f"Predominant whale selling: {buys}/{total}"  
    return {"score": score, "details": details}


def analyze_sentiment(news_items: list) -> dict:
    if not news_items:
        return {"score": 5, "details": "No news data"}
    neg_kw = ["rug pull", "hack", "fraud", "downturn"]
    neg = sum(1 for n in news_items if any(w in (n.title or "").lower() for w in neg_kw))
    total = len(news_items)
    if neg / total > 0.3:
        score, details = 2, f"{neg}/{total} negative headlines"
    elif neg > 0:
        score, details = 5, f"{neg}/{total} negative headlines"
    else:
        score, details = 8, "Mostly positive headlines"
    return {"score": score, "details": details}


def analyze_risk_reward(prices: list) -> dict:
    if not prices or len(prices) < 10:
        return {"score": 5, "details": "Insufficient data for volatility"}
    sorted_p = sorted(prices, key=lambda p: p.time)
    closes = [p.close for p in sorted_p if p.close is not None]
    returns = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev > 0:
            returns.append((closes[i] - prev) / prev)
    if not returns:
        return {"score": 5, "details": "No valid returns"}
    stdev = statistics.pstdev(returns)
    if stdev < 0.01:
        score, details = 8, f"Low vol {stdev:.2%}"  
    elif stdev < 0.02:
        score, details = 6, f"Moderate vol {stdev:.2%}"  
    else:
        score, details = 3, f"High vol {stdev:.2%}"  
    return {"score": score, "details": details}


def analyze_druckenmiller_valuation(financial_line_items: list, market_cap: float | None) -> dict:
    """
    Druckenmiller is willing to pay up for growth, but still checks:
      - P/E
      - P/FCF
      - EV/EBIT
      - EV/EBITDA
    Each can yield up to 2 points => max 8 raw points => scale to 0–10.
    """
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    details = []
    raw_score = 0

    # Gather needed data
    net_incomes = [fi.net_income for fi in financial_line_items if fi.net_income is not None]
    fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]
    ebit_values = [fi.ebit for fi in financial_line_items if fi.ebit is not None]
    ebitda_values = [fi.ebitda for fi in financial_line_items if fi.ebitda is not None]

    # For EV calculation, let's get the most recent total_debt & cash
    debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
    cash_values = [fi.cash_and_equivalents for fi in financial_line_items if fi.cash_and_equivalents is not None]
    recent_debt = debt_values[0] if debt_values else 0
    recent_cash = cash_values[0] if cash_values else 0

    enterprise_value = market_cap + recent_debt - recent_cash

    # 1) P/E
    recent_net_income = net_incomes[0] if net_incomes else None
    if recent_net_income and recent_net_income > 0:
        pe = market_cap / recent_net_income
        pe_points = 0
        if pe < 15:
            pe_points = 2
            details.append(f"Attractive P/E: {pe:.2f}")
        elif pe < 25:
            pe_points = 1
            details.append(f"Fair P/E: {pe:.2f}")
        else:
            details.append(f"High or Very high P/E: {pe:.2f}")
        raw_score += pe_points
    else:
        details.append("No positive net income for P/E calculation")

    # 2) P/FCF
    recent_fcf = fcf_values[0] if fcf_values else None
    if recent_fcf and recent_fcf > 0:
        pfcf = market_cap / recent_fcf
        pfcf_points = 0
        if pfcf < 15:
            pfcf_points = 2
            details.append(f"Attractive P/FCF: {pfcf:.2f}")
        elif pfcf < 25:
            pfcf_points = 1
            details.append(f"Fair P/FCF: {pfcf:.2f}")
        else:
            details.append(f"High/Very high P/FCF: {pfcf:.2f}")
        raw_score += pfcf_points
    else:
        details.append("No positive free cash flow for P/FCF calculation")

    # 3) EV/EBIT
    recent_ebit = ebit_values[0] if ebit_values else None
    if enterprise_value > 0 and recent_ebit and recent_ebit > 0:
        ev_ebit = enterprise_value / recent_ebit
        ev_ebit_points = 0
        if ev_ebit < 15:
            ev_ebit_points = 2
            details.append(f"Attractive EV/EBIT: {ev_ebit:.2f}")
        elif ev_ebit < 25:
            ev_ebit_points = 1
            details.append(f"Fair EV/EBIT: {ev_ebit:.2f}")
        else:
            details.append(f"High EV/EBIT: {ev_ebit:.2f}")
        raw_score += ev_ebit_points
    else:
        details.append("No valid EV/EBIT because EV <= 0 or EBIT <= 0")

    # 4) EV/EBITDA
    recent_ebitda = ebitda_values[0] if ebitda_values else None
    if enterprise_value > 0 and recent_ebitda and recent_ebitda > 0:
        ev_ebitda = enterprise_value / recent_ebitda
        ev_ebitda_points = 0
        if ev_ebitda < 10:
            ev_ebitda_points = 2
            details.append(f"Attractive EV/EBITDA: {ev_ebitda:.2f}")
        elif ev_ebitda < 18:
            ev_ebitda_points = 1
            details.append(f"Fair EV/EBITDA: {ev_ebitda:.2f}")
        else:
            details.append(f"High EV/EBITDA: {ev_ebitda:.2f}")
        raw_score += ev_ebitda_points
    else:
        details.append("No valid EV/EBITDA because EV <= 0 or EBITDA <= 0")

    # We have up to 2 points for each of the 4 metrics => 8 raw points max
    # Scale raw_score to 0–10
    final_score = min(10, (raw_score / 8) * 10)

    return {"score": final_score, "details": "; ".join(details)}


def generate_druckenmiller_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> StanleyDruckenmillerSignal:
    """
    Generates a JSON signal in the style of crypto-adapted Stanley Druckenmiller.
    """
    template = ChatPromptTemplate.from_messages([
        ("system",
         """
You are a crypto-focused Stanley Druckenmiller AI agent, making investment decisions for digital assets using these principles:

1. Seek asymmetric risk-reward: large upside, controlled downside.
2. Emphasize price momentum and on-chain adoption metrics.
3. Factor in network fundamentals: active addresses, transaction volume, dev activity.
4. Incorporate tokenomics: circulating supply changes, staking yields, inflation.
5. Analyze whale transaction flows for institutional signals.
6. Evaluate crypto market sentiment: news headlines and social trends.
7. Preserve capital by avoiding extreme volatility and drawdowns.
8. Be aggressive when conviction and on-chain signals align.
9. Cut losses quickly when momentum and fundamentals falter.

Rules:
- Reward tokens with strong momentum and improving on-chain metrics.
- Penalize assets with high volatility or negative on-chain signals.
- Reference specific numeric values from analysis_data in your reasoning.
- Output strictly JSON: {"signal":"bullish/bearish/neutral","confidence":float,"reasoning":"string"}

When reasoning:
- Cite price change percentages and volatility stats.
- Mention whale buy/sell ratios.
- Highlight key on-chain metrics (address growth, volume).
- Address macro catalysts or regulatory risks.
- Use a decisive, conviction-driven tone.

Example bullish reasoning:
"The token rallied 35% over the past month on 60% increase in active addresses. Daily volatility is low at 1.2% stdev, and whale flows show net buys of 80% of transactions. News sentiment is bullish around upcoming network upgrade. Risk-reward remains asymmetric with 70% upside potential vs 15% downside given strong tokenomics."

Example bearish reasoning:
"Despite 20% rally, active addresses declined by 10% and daily volatility spiked to 4.5% stdev. Whale trades show 70% net sells, and negative headlines around regulatory scrutiny persist. Downside risk of 40% outweighs limited upside."
"""),
        ("human",
         """
Based on the analysis data for {ticker}:
{analysis_data}

Return the signal in JSON:
{"signal":"bullish/bearish/neutral","confidence":float,"reasoning":"string"}
"""),
    ])
    prompt = template.invoke({"ticker": ticker, "analysis_data": json.dumps(analysis_data, indent=2)})
    def default():
        return StanleyDruckenmillerSignal(signal="neutral", confidence=0.0, reasoning="Error; default neutral")
    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=StanleyDruckenmillerSignal,
        agent_name="stanley_druckenmiller_agent",
        default_factory=default,
    )