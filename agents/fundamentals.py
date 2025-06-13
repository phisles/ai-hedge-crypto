from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import json

from tools.api import get_financial_metrics

##### Fundamental Agent #####
def fundamentals_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    fundamental_analysis = {}

    for ticker in tickers:
        progress.update_status("fundamentals_agent", ticker, "Fetching financial metrics")
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=10,
        )
        if not financial_metrics:
            progress.update_status("fundamentals_agent", ticker, "Failed: No financial metrics found")
            continue

        metrics = financial_metrics[0]

        # —— CRYPTO BRANCH START —— 
        if ticker.upper().endswith(("/USD", "-USD")):
            price_30d = metrics.get("price_change_pct_30d", 0.0)
            sentiment = metrics.get("sentiment_votes_up_pct", 0.0)
            vol_mc = metrics.get("volume_to_market_cap", 0.0)

            score = 0
            if price_30d > 0.10:
                score += 2
            elif price_30d < -0.10:
                score -= 1
            if sentiment > 60:
                score += 1
            if vol_mc > 0.02:
                score += 1

            if score >= 3:
                sig = "bullish"
            elif score <= 0:
                sig = "bearish"
            else:
                sig = "neutral"

            conf = round(min(max((score + 1) / 4, 0), 1) * 100)
            fundamental_analysis[ticker] = {
                "signal": sig,
                "confidence": conf,
                "reasoning": (
                    f"30d Δ {price_30d:.1%}, sentiment {sentiment:.1f}%, "
                    f"vol/MC {vol_mc:.1%}"
                )
            }
            progress.update_status("fundamentals_agent", ticker, "Done (crypto)")
            continue
        # —— CRYPTO BRANCH END —— 

        # Initialize equity signals
        signals = []
        reasoning = {}

        progress.update_status("fundamentals_agent", ticker, "Analyzing profitability")
        return_on_equity = metrics.return_on_equity
        net_margin = metrics.net_margin
        operating_margin = metrics.operating_margin
        thresholds = [
            (return_on_equity, 0.15),
            (net_margin, 0.20),
            (operating_margin, 0.15),
        ]
        profitability_score = sum(m is not None and m > t for m, t in thresholds)
        signals.append(
            "bullish" if profitability_score >= 2 
            else "bearish" if profitability_score == 0 
            else "neutral"
        )
        reasoning["profitability_signal"] = {
            "signal": signals[0],
            "details": (
                (f"ROE: {return_on_equity:.2%}" if return_on_equity is not None else "ROE: N/A")
                + ", "
                + (f"Net Margin: {net_margin:.2%}" if net_margin is not None else "Net Margin: N/A")
                + ", "
                + (f"Op Margin: {operating_margin:.2%}" if operating_margin is not None else "Op Margin: N/A")
            ),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing growth")
        revenue_growth = metrics.revenue_growth
        earnings_growth = metrics.earnings_growth
        book_value_growth = metrics.book_value_growth
        thresholds = [
            (revenue_growth, 0.10),
            (earnings_growth, 0.10),
            (book_value_growth, 0.10),
        ]
        growth_score = sum(m is not None and m > t for m, t in thresholds)
        signals.append(
            "bullish" if growth_score >= 2 
            else "bearish" if growth_score == 0 
            else "neutral"
        )
        reasoning["growth_signal"] = {
            "signal": signals[1],
            "details": (
                (f"Revenue Growth: {revenue_growth:.2%}" if revenue_growth is not None else "Revenue Growth: N/A")
                + ", "
                + (f"Earnings Growth: {earnings_growth:.2%}" if earnings_growth is not None else "Earnings Growth: N/A")
            ),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing financial health")
        current_ratio = metrics.current_ratio
        debt_to_equity = metrics.debt_to_equity
        free_cash_flow_per_share = metrics.free_cash_flow_per_share
        earnings_per_share = metrics.earnings_per_share
        health_score = 0
        if current_ratio is not None and current_ratio > 1.5:
            health_score += 1
        if debt_to_equity is not None and debt_to_equity < 0.5:
            health_score += 1
        if (
            free_cash_flow_per_share is not None and
            earnings_per_share is not None and
            free_cash_flow_per_share > earnings_per_share * 0.8
        ):
            health_score += 1
        signals.append(
            "bullish" if health_score >= 2 
            else "bearish" if health_score == 0 
            else "neutral"
        )
        reasoning["financial_health_signal"] = {
            "signal": signals[2],
            "details": (
                (f"Current Ratio: {current_ratio:.2f}" if current_ratio is not None else "Current Ratio: N/A")
                + ", "
                + (f"D/E: {debt_to_equity:.2f}" if debt_to_equity is not None else "D/E: N/A")
            ),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing valuation ratios")
        pe_ratio = metrics.price_to_earnings_ratio
        pb_ratio = metrics.price_to_book_ratio
        ps_ratio = metrics.price_to_sales_ratio
        thresholds = [
            (pe_ratio, 25),
            (pb_ratio, 3),
            (ps_ratio, 5),
        ]
        price_ratio_score = sum(m is not None and m > t for m, t in thresholds)
        signals.append(
            "bearish" if price_ratio_score >= 2 
            else "bullish" if price_ratio_score == 0 
            else "neutral"
        )
        reasoning["price_ratios_signal"] = {
            "signal": signals[3],
            "details": (
                (f"P/E: {pe_ratio:.2f}" if pe_ratio is not None else "P/E: N/A")
                + ", "
                + (f"P/B: {pb_ratio:.2f}" if pb_ratio is not None else "P/B: N/A")
                + ", "
                + (f"P/S: {ps_ratio:.2f}" if ps_ratio is not None else "P/S: N/A")
            ),
        }

        progress.update_status("fundamentals_agent", ticker, "Calculating final signal")
        bullish_signals = signals.count("bullish")
        bearish_signals = signals.count("bearish")
        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        total_signals = len(signals)
        confidence = round(max(bullish_signals, bearish_signals) / total_signals, 2) * 100

        fundamental_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("fundamentals_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(fundamental_analysis),
        name="fundamentals_agent",
    )

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")

    state["data"]["analyst_signals"]["fundamentals_agent"] = fundamental_analysis

    return {
        "messages": [message],
        "data": data,
    }