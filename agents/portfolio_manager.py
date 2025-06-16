import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "hold"]
    quantity: float = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence in the decision, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the decision")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders for multiple tickers"""

    # Get the portfolio and analyst signals
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]
    print("\n🧪 DEBUG: FULL state['data']['max_shares'] before decision logic:")
    for t, v in state["data"].get("max_shares", {}).items():
        print(f"  {t}: {v}")

    progress.update_status("portfolio_management_agent", None, "Analyzing signals")

    # Get position limits, current prices, and signals for every ticker
    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status("portfolio_management_agent", ticker, "Processing analyst signals")

        # Get position limits and current prices for the ticker
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)
        current_prices[ticker] = risk_data.get("current_price", 0)

        # Calculate maximum shares allowed based on position limit and price
        # Use precomputed max_shares directly from risk_management_agent output
        print(f"📥 FETCHED max_shares for {ticker}: {state['data'].get('max_shares', {}).get(ticker)}")
        max_shares[ticker] = state["data"].get("max_shares", {}).get(ticker, {
            "long": 0,
            "short": 0,
        })

        # Get signals for the ticker
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if agent != "risk_management_agent" and ticker in signals:
                ticker_signals[agent] = {"signal": signals[ticker]["signal"], "confidence": signals[ticker]["confidence"]}
        signals_by_ticker[ticker] = ticker_signals

    progress.update_status("portfolio_management_agent", None, "Making trading decisions")
    # 🔍 Debug max shares, limits, and prices
    print("\n📊 RISK-CHECKED MAX SHARES:")
    for t in tickers:
        print(f"🔹 {t}: max_shares={max_shares[t]}, limit=${position_limits[t]:.2f}, price=${current_prices[t]:.2f}")
    # Generate the trading decision
    print("\n🧾 max_shares passed to LLM:")
    for t in tickers:
        print(f"  {t}: {max_shares[t]}")
    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        model_name=state["metadata"]["model_name"],
        model_provider=state["metadata"]["model_provider"],
    )

    # Create the portfolio management message
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name="portfolio_management",
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "Portfolio Management Agent")

    progress.update_status("portfolio_management_agent", None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, dict[str, int]],
    portfolio: dict[str, float],
    model_name: str,
    model_provider: str,
) -> PortfolioManagerOutput:
    """Attempts to get a decision from the LLM with retry logic"""

    # Create the prompt template           
    template = ChatPromptTemplate.from_messages([
        ("system",
        """You are a portfolio manager making final trading decisions for a set of tickers.
    
    Your job is to choose the best action per ticker: 'buy', 'sell', or 'hold'.
    
    Use these guidelines:
    
    BUY:
    - You may buy even if not all signals are bullish — a strong confidence from 1–2 agents can be enough.
    - Use `max_shares[ticker]["long"]` as a ceiling for quantity.
    - Adjust the quantity up or down based on confidence:
      * Very confident (≥80): buy near max
      * Moderate (50–80): buy partial
      * Low confidence (30–50): small exploratory buy
      * Very low (<30): usually avoid buying
    
    SELL:
    - Be cautious with sells.
    - Do **not** sell just because there is a bearish majority.
    - Only sell if:
      * Bearish signals are strong **and** confident, **and**
      * You currently hold a position in that ticker.
    - You can sell part or all of the position based on confidence:
      * Strong: full exit
      * Moderate: reduce size
      * Weak: consider holding
    
    HOLD:
    - Appropriate when signals are mixed or unclear.
    - Also valid if confidence in buy/sell is low.
    
    ALWAYS:
    - Factor in current portfolio positions when deciding.
    - Use `portfolio_cash`, `portfolio_positions`, and `max_shares` responsibly.
    - Explain your reasoning clearly.
    
    Output format (respond only with valid JSON):
    {{  
      "decisions": {{  
        "TICKER": {{  
          "action": "buy"|"sell"|"hold",  
          "quantity": float,  
          "confidence": float (0–100),  
          "reasoning": string  
        }},
        ...
      }}
    }}
    
    Respond only with a valid JSON object.
    """)
    ])

    # Generate the prompt
    relative_confidence = {
        ticker: sum(s["confidence"] for s in signals.values() if s["signal"] == "bullish")
        for ticker, signals in signals_by_ticker.items()
        if sum(1 for s in signals.values() if s["signal"] == "bullish") >= 2
    }

    prompt = template.invoke(
        {
            "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),
            "current_prices": json.dumps(current_prices, indent=2),
            "max_shares": json.dumps({t: {"long": max_shares[t]["long"]} for t in tickers}, indent=2),
            "portfolio_cash": f"{portfolio.get('cash', 0):.2f}",
            "portfolio_positions": json.dumps(
                {k: v["long"] for k, v in portfolio.get("positions", {}).items() if v["long"] > 0}, indent=2
            ),
            "relative_confidence": json.dumps(relative_confidence, indent=2),
        }
    )

    # Create default factory for PortfolioManagerOutput
    def create_default_portfolio_output():
        return PortfolioManagerOutput(
            decisions={
                ticker: PortfolioDecision(
                    action="hold",
                    quantity=0.0,
                    confidence=0.0,
                    reasoning="Error in portfolio management, defaulting to hold"
                )
                for ticker in tickers
            }
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=PortfolioManagerOutput,
        agent_name="portfolio_management_agent",
        default_factory=create_default_portfolio_output
    )
