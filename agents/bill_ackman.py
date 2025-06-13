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
from data.models import LineItem


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
        metrics = get_financial_metrics(ticker, end_date)
        
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

    # ─── DEBUG: show exactly which fields each LineItem has ─────────────────────
    for item in financial_line_items:
        print(f"🔍 {item.ticker} payload keys:", item.__dict__.keys())

    # 1. Multi-period revenue growth analysis
    revenues = [
        getattr(item, "revenue", None)
        for item in financial_line_items
        if getattr(item, "revenue", None) is not None
    ]
    if len(revenues) >= 2:
        initial, final = revenues[0], revenues[-1]
        if initial and final and final > initial:
            growth_rate = (final - initial) / abs(initial)
            if growth_rate > 0.5:
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
    fcf_vals = [
        getattr(item, "free_cash_flow", None)
        for item in financial_line_items
        if getattr(item, "free_cash_flow", None) is not None
    ]
    op_margin_vals = [
        getattr(item, "operating_margin", None)
        for item in financial_line_items
        if getattr(item, "operating_margin", None) is not None
    ]

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
    latest = metrics[0]
    if isinstance(latest, dict):
        ps_ratio  = latest.get("price_to_sales_ratio")
        dev_stars = latest.get("developer_stars")
    else:
        ps_ratio  = getattr(latest, "price_to_sales_ratio", None)
        dev_stars = getattr(latest, "developer_stars", None)

    if isinstance(ps_ratio, (int, float)) and ps_ratio < 50:
        score += 1
        details.append(f"Healthy price-to-sales ratio: {ps_ratio:.2f}")
    else:
        details.append("No P/S data or ratio is too high.")

    # Developer activity
    if isinstance(dev_stars, (int, float)) and dev_stars > 5000:
        score += 1
        details.append(f"Strong developer support: {dev_stars} GitHub stars.")
    else:
        details.append("Weak or unknown developer activity.")

    # 4. (Optional) Brand Intangible (if intangible_assets are fetched)
    # intangible_vals = [getattr(item, "intangible_assets", 0) for item in financial_line_items]
    # if any(intangible_vals):
    #     score += 1
    #     details.append("Significant intangible assets may indicate brand value or proprietary tech.")

    return {
        "score": score,
        "details": "; ".join(details)
    }


def analyze_financial_discipline(metrics: list[dict], financial_line_items: list) -> dict:
    """
    Assess the project’s financial discipline — inflation control (circulating vs total supply),
    leverage (debt vs assets), and other balance‐sheet metrics.
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze financial discipline"
        }

    # latest metrics is a dict
    latest = metrics[0]
    # DEBUG: show what keys we actually have
    print("🔍 Latest metrics keys:", list(latest.keys()))

    # 1) Inflation control: circulating_supply / total_supply
    circ = latest.get("circulating_supply")
    tot  = latest.get("total_supply")
    if isinstance(circ, (int, float)) and isinstance(tot, (int, float)) and tot > 0:
        ratio = circ / tot
        print(f"🔍 Supply ratio circ/total: {ratio:.3f}")
        if ratio < 0.7:
            score += 1
            details.append(f"Low inflation: only {ratio:.0%} of tokens are circulating.")
        else:
            details.append(f"High inflation: {ratio:.0%} circulating.")
    else:
        details.append("No supply data available for inflation check.")

    # 2) Leverage: look for 'debt' and 'assets' on the first line item
    sample_li = financial_line_items[0]
    # DEBUG: show which dynamic fields were loaded on LineItem
    print("🔍 LineItem fields:", sample_li.__dict__.keys())

    debt   = sample_li.__dict__.get("debt")
    assets = sample_li.__dict__.get("assets")
    if isinstance(debt, (int, float)) and isinstance(assets, (int, float)) and assets > 0:
        lev = debt / assets
        print(f"🔍 Debt/assets ratio: {lev:.3f}")
        if lev < 0.5:
            score += 1
            details.append(f"Conservative leverage: debt/assets {lev:.2f}.")
        else:
            details.append(f"High leverage: debt/assets {lev:.2f}.")
    else:
        details.append("No debt/assets data available.")

    # (You can add more checks here...)

    return {
        "score": score,
        "details": "; ".join(details)
    }


# at top of agents/bill_ackman.py
from data.models import LineItem

def analyze_activism_potential(
    financial_line_items: list[LineItem],
    metrics: list[dict],
) -> dict:
    """
    Estimate how attractive the company is as an activist target by
    looking for undervaluation, under‐performing assets, or activist‐friendly
    balance‐sheet setups.
    """
    score = 0
    details = []

    # 1. Under‐performance vs peers: look at 60d and 1y price change
    if metrics:
        latest = metrics[0]
        pct60 = latest.get("price_change_pct_60d")
        pct1y = latest.get("price_change_pct_1y")
        if isinstance(pct60, (int, float)) and pct60 < 0:
            score += 1
            details.append(f"Price down {pct60*100:.1f}% over 60d (potential undervaluation).")
        else:
            details.append("No 60d underperformance signal.")
        if isinstance(pct1y, (int, float)) and pct1y < 0:
            score += 1
            details.append(f"Price down {pct1y*100:.1f}% over 1y (longer‐term malaise).")
        else:
            details.append("No 1y underperformance signal.")
    else:
        details.append("No market‐data metrics available for activism check.")

    # 2. Balance sheet liquidity: current ratio from the latest line‐item
    if financial_line_items:
        latest_li = financial_line_items[-1]
        if isinstance(latest_li, dict):
            cr = latest_li.get("current_ratio")
        else:
            cr = getattr(latest_li, "current_ratio", None)

        if isinstance(cr, (int, float)):
            if cr > 1.5:
                score += 1
                details.append(f"Strong liquidity (current ratio {cr:.2f}).")
            else:
                details.append(f"Modest liquidity (current ratio {cr:.2f}).")
        else:
            details.append("No current ratio available.")
    else:
        details.append("No balance‐sheet line items for liquidity check.")

    # 3. Leverage: debt/assets from the latest line‐item
    if financial_line_items:
        latest_li = financial_line_items[-1]
        if isinstance(latest_li, dict):
            debt = latest_li.get("debt")
            assets = latest_li.get("assets")
        else:
            debt = getattr(latest_li, "debt", None)
            assets = getattr(latest_li, "assets", None)

        if isinstance(debt, (int, float)) and isinstance(assets, (int, float)) and assets > 0:
            lev = debt / assets
            if lev < 0.5:
                score += 1
                details.append(f"Conservative leverage (debt/assets {lev:.2f}).")
            else:
                details.append(f"High leverage (debt/assets {lev:.2f}).")
        else:
            details.append("No debt/assets data for leverage check.")
    # if no financial_line_items, already covered above

    return {
        "score": score,
        "details": "; ".join(details)
    }

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