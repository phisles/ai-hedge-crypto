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


class BillAckmanSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def bill_ackman_agent(state: AgentState):
    """
    Analyzes stocks using Bill Ackman's investing principles and LLM reasoning.
    Fetches multiple periods of data for a more robust long-term view.
    Incorporates brand/competitive advantage, activism potential, and other key factors.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    
    analysis_data = {}
    ackman_analysis = {}
    
    for ticker in tickers:
        progress.update_status("bill_ackman_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)
        
        progress.update_status("bill_ackman_agent", ticker, "Gathering financial line items")
        # Request multiple periods of data (annual or TTM) for a more robust long-term view.
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                # Optional: intangible_assets if available
                # "intangible_assets"
            ],
            end_date,
            period="annual",
            limit=5
        )
        
        progress.update_status("bill_ackman_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)
        
        progress.update_status("bill_ackman_agent", ticker, "Analyzing business quality")
        quality_analysis = analyze_business_quality(metrics, financial_line_items)
        
        progress.update_status("bill_ackman_agent", ticker, "Analyzing balance sheet and capital structure")
        balance_sheet_analysis = analyze_financial_discipline(metrics, financial_line_items)
        
        progress.update_status("bill_ackman_agent", ticker, "Analyzing activism potential")
        activism_analysis = analyze_activism_potential(financial_line_items, metrics)
        
        progress.update_status("bill_ackman_agent", ticker, "Calculating intrinsic value & margin of safety")
        valuation_analysis = analyze_valuation(financial_line_items, market_cap)
        
        # Combine partial scores or signals
        total_score = (
            quality_analysis["score"]
            + balance_sheet_analysis["score"]
            + activism_analysis["score"]
            + valuation_analysis["score"]
        )
        max_possible_score = 20  # Adjust weighting as desired (5 from each sub-analysis, for instance)
        
        # Generate a simple buy/hold/sell (bullish/neutral/bearish) signal
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
            "quality_analysis": quality_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "activism_analysis": activism_analysis,
            "valuation_analysis": valuation_analysis
        }
        
        progress.update_status("bill_ackman_agent", ticker, "Generating Bill Ackman analysis")
        ackman_output = generate_ackman_output(
            ticker=ticker, 
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        
        ackman_analysis[ticker] = {
            "signal": ackman_output.signal,
            "confidence": ackman_output.confidence,
            "reasoning": ackman_output.reasoning
        }
        
        progress.update_status("bill_ackman_agent", ticker, "Done")
    
    # Wrap results in a single message for the chain
    message = HumanMessage(
        content=json.dumps(ackman_analysis),
        name="bill_ackman_agent"
    )
    
    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(ackman_analysis, "Bill Ackman Agent")
    
    # Add signals to the overall state
    state["data"]["analyst_signals"]["bill_ackman_agent"] = ackman_analysis

    return {
        "messages": [message],
        "data": state["data"]
    }


def analyze_business_quality(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze whether the company has a high-quality business with stable or growing cash flows,
    durable competitive advantages (moats), and potential for long-term growth.
    Also tries to infer brand strength if intangible_assets data is present (optional).
    """
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze business quality"
        }
    
    # 1. Multi-period revenue growth analysis
    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    if len(revenues) >= 2:
        initial, final = revenues[0], revenues[-1]
        if initial and final and final > initial:
            growth_rate = (final - initial) / abs(initial)
            if growth_rate > 0.5:  # e.g., 50% cumulative growth
                score += 2
                details.append(f"Revenue grew by {(growth_rate*100):.1f}% over the full period (strong growth).")
            else:
                score += 1
                details.append(f"Revenue growth is positive but under 50% cumulatively ({(growth_rate*100):.1f}%).")
        else:
            details.append("Revenue did not grow significantly or data insufficient.")
    else:
        details.append("Not enough revenue data for multi-period trend.")
    
    # 2. Operating margin and free cash flow consistency
    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]
    
    if op_margin_vals:
        above_15 = sum(1 for m in op_margin_vals if m > 0.15)
        if above_15 >= (len(op_margin_vals) // 2 + 1):
            score += 2
            details.append("Operating margins have often exceeded 15% (indicates good profitability).")
        else:
            details.append("Operating margin not consistently above 15%.")
    else:
        details.append("No operating margin data across periods.")
    
    if fcf_vals:
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
        if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
            score += 1
            details.append("Majority of periods show positive free cash flow.")
        else:
            details.append("Free cash flow not consistently positive.")
    else:
        details.append("No free cash flow data across periods.")
    
    # 3. Return on Equity (ROE) check from the latest metrics
    latest_metrics = metrics[0]
    if latest_metrics.price_to_sales_ratio and isinstance(latest_metrics.price_to_sales_ratio, (int, float)) and latest_metrics.price_to_sales_ratio < 50:
        score += 1
        details.append(f"Healthy price-to-sales ratio: {latest_metrics.price_to_sales_ratio:.2f}")
    else:
        details.append("No P/S data or ratio is too high.")
    # Developer activity
    if latest_metrics.developer_stars and latest_metrics.developer_stars > 5000:
        score += 1
        details.append(f"Strong developer support: {latest_metrics.developer_stars} GitHub stars.")
    else:
        details.append("Weak or unknown developer activity.")
    
    # 4. (Optional) Brand Intangible (if intangible_assets are fetched)
    # intangible_vals = [item.intangible_assets for item in financial_line_items if item.intangible_assets]
    # if intangible_vals and sum(intangible_vals) > 0:
    #     details.append("Significant intangible assets may indicate brand value or proprietary tech.")
    #     score += 1
    
    return {
        "score": score,
        "details": "; ".join(details)
    }


def analyze_financial_discipline(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    latest = financial_line_items[-1] if financial_line_items else None
    if not latest or not metrics:
        return {"score": 0, "details": "Insufficient data for crypto financial discipline"}

    # Token supply growth (less inflation = better)
    if latest.circulating_supply and latest.total_supply and latest.total_supply > 0:
        inflation_rate = (latest.total_supply - latest.circulating_supply) / latest.total_supply
        if inflation_rate < 0.05:
            score += 1
            details.append(f"Low inflation: only {inflation_rate*100:.2f}% of tokens remain uncirculated.")
        else:
            details.append(f"Token inflation may be high ({inflation_rate*100:.2f}% remaining).")

    # Price stability check (rough volatility proxy)
    if metrics[0].price_change_pct_60d and abs(metrics[0].price_change_pct_60d) < 20:
        score += 1
        details.append(f"Moderate 60-day price volatility: {metrics[0].price_change_pct_60d:.2f}%")
    else:
        details.append("High 60-day price volatility.")

    return {"score": score, "details": "; ".join(details)}


def analyze_activism_potential(financial_line_items: list, metrics: list) -> dict:
    """
    In crypto, we interpret 'activism potential' as:
    - Short-term weakness despite strong long-term trend.
    - Potential for catalysts like upgrades, listings, or improved tokenomics.
    """

    if not metrics or len(metrics) == 0:
        return {
            "score": 0,
            "details": "No metrics available for activism-style signal."
        }

    m = metrics[0]  # Use most recent

    long_term = m.price_change_pct_60d
    short_term = m.price_change_pct_7d

    score = 0
    details = []

    if long_term and short_term:
        if long_term > 10 and short_term < -2:
            score += 2
            details.append(
                f"Strong long-term trend ({long_term:.1f}%) with recent pullback ({short_term:.1f}%) — possible reentry/catalyst zone."
            )
        else:
            details.append("No significant divergence between long- and short-term price action.")
    else:
        details.append("Missing 60d or 7d price data.")

    return {"score": score, "details": "; ".join(details)}


def analyze_valuation(financial_line_items: list, market_cap: float) -> dict:
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "No data to value crypto asset"}

    latest = financial_line_items[-1]
    volume = latest.volume_24h if hasattr(latest, "volume_24h") else None

    if volume and market_cap > 0:
        ratio = volume / market_cap
        score = 1 if ratio > 0.05 else 0
        return {
            "score": score,
            "details": f"Volume-to-market cap ratio: {ratio:.2%} — {'healthy' if score else 'low'} liquidity."
        }
    return {"score": 0, "details": "Volume or market cap data missing"}


def generate_ackman_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> BillAckmanSignal:
    """
    Generates investment decisions in the style of Bill Ackman.
    Includes more explicit references to brand strength, activism potential, 
    catalysts, and management changes in the system prompt.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Bill Ackman-inspired crypto analyst. Your job is to evaluate cryptocurrency assets using principles adapted from activist investing.

            1. Seek high-quality crypto projects with strong developer communities, network adoption, and utility-based ecosystems.
            2. Prioritize long-term growth potential, robust market presence, and healthy tokenomics (e.g. controlled supply, strong demand).
            3. Reward disciplined supply management (low inflation, fixed caps, low dilution).
            4. Valuation matters: consider metrics like market cap, trading volume, and price-to-sales ratio.
            5. Look for activist-style opportunities: assets showing long-term strength but short-term weakness or inefficiencies that could be corrected (e.g., underused upgrades, poor listing exposure, or unoptimized token design).
            6. Focus on a few high-conviction ideas where value can be unlocked.

            In your reasoning:
            - Emphasize strong network effects, developer activity, and real-world usage.
            - Review supply inflation, price trends, and volume-to-market-cap ratios as indicators of sustainability.
            - Use GitHub activity and release momentum to gauge development health.
            - Avoid or penalize excessive volatility, centralized control, or unclear utility.
            - Provide a valuation assessment with quantitative support (e.g., P/S ratio, liquidity, market dominance).
            - Identify any catalysts like protocol upgrades, exchange listings, or upcoming ecosystem expansions.
            - Use a confident, analytical tone when justifying ratings, and don’t hesitate to flag concerns.

            Return your final recommendation (signal: bullish, neutral, or bearish) with a 0-100 confidence score and a detailed explanation.
            """
        ),
        (
            "human",
            """Based on the following analysis, create an Ackman-style investment signal.

            Analysis Data for {ticker}:
            {analysis_data}

            Return your output in strictly valid JSON:
            {{
              "signal": "bullish" | "bearish" | "neutral",
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

    def create_default_bill_ackman_signal():
        return BillAckmanSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt, 
        model_name=model_name, 
        model_provider=model_provider, 
        pydantic_model=BillAckmanSignal, 
        agent_name="bill_ackman_agent", 
        default_factory=create_default_bill_ackman_signal,
    )