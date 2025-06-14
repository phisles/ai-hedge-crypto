import math
import json
import pandas as pd
import numpy as np

from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_prices, prices_to_df, get_financial_metrics
from utils.progress import progress


##### Technical Analyst #####
def technical_analyst_agent(state: AgentState):
    """
    Sophisticated technical analysis system that combines multiple trading strategies for multiple tickers:
    1. Trend Following
    2. Mean Reversion
    3. Momentum (with network volume confirmation)
    4. Volatility Analysis
    5. Statistical Arbitrage Signals
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    technical_analysis = {}

    for ticker in tickers:
        progress.update_status("technical_analyst_agent", ticker, "Analyzing price data")

        # ─── Price history ────────────────────────────────────────────────────────
        prices = get_prices(ticker=ticker, start_date=start_date, end_date=end_date)
        if not prices:
            progress.update_status("technical_analyst_agent", ticker, "Failed: No price data found")
            continue
        prices_df = prices_to_df(prices)

        # ─── On-chain network volume ──────────────────────────────────────────────
        metrics = get_financial_metrics(ticker, end_date)
        network_vol = metrics[0].get("total_volume", 0) if isinstance(metrics, list) and metrics else 0

        progress.update_status("technical_analyst_agent", ticker, "Calculating trend signals")
        trend_signals = calculate_trend_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Calculating mean reversion")
        mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Calculating momentum")
        momentum_signals = calculate_momentum_signals(prices_df, network_vol)

        progress.update_status("technical_analyst_agent", ticker, "Analyzing volatility")
        volatility_signals = calculate_volatility_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Statistical analysis")
        stat_arb_signals = calculate_stat_arb_signals(prices_df)

        # ─── Combine all signals using a weighted ensemble approach ──────────────
        strategy_weights = {
            "trend": 0.20,
            "mean_reversion": 0.15,
            "momentum": 0.30,
            "volatility": 0.20,
            "stat_arb": 0.15,
        }

        progress.update_status("technical_analyst_agent", ticker, "Combining signals")
        combined_signal = weighted_signal_combination(
            {
                "trend": trend_signals,
                "mean_reversion": mean_reversion_signals,
                "momentum": momentum_signals,
                "volatility": volatility_signals,
                "stat_arb": stat_arb_signals,
            },
            strategy_weights,
        )

        technical_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": round(combined_signal["confidence"] * 100),
            "strategy_signals": {
                "trend_following": {
                    "signal": trend_signals["signal"],
                    "confidence": round(trend_signals["confidence"] * 100),
                    "metrics": normalize_pandas(trend_signals["metrics"]),
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": round(mean_reversion_signals["confidence"] * 100),
                    "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": round(momentum_signals["confidence"] * 100),
                    "metrics": normalize_pandas(momentum_signals["metrics"]),
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": round(volatility_signals["confidence"] * 100),
                    "metrics": normalize_pandas(volatility_signals["metrics"]),
                },
                "statistical_arbitrage": {
                    "signal": stat_arb_signals["signal"],
                    "confidence": round(stat_arb_signals["confidence"] * 100),
                    "metrics": normalize_pandas(stat_arb_signals["metrics"]),
                },
            },
        }
        print(f"📤 Technical Analyst output for {ticker}:\n", json.dumps(technical_analysis[ticker], indent=2, default=str))
        progress.update_status("technical_analyst_agent", ticker, "Done")

    # ←— updated to stringify any non-JSON-serializable types (like bools)
    message = HumanMessage(
        content=json.dumps(technical_analysis, default=str),
        name="technical_analyst_agent",
    )
    
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(technical_analysis, "Technical Analyst")

    state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def calculate_trend_signals(prices_df: pd.DataFrame) -> dict:
    """
    Advanced trend following strategy using multiple timeframes and indicators
    """
    ema_12 = calculate_ema(prices_df, 12)
    ema_26 = calculate_ema(prices_df, 26)
    ema_50 = calculate_ema(prices_df, 50)

    adx = calculate_adx(prices_df, 14)
    trend_strength = adx["adx"].iloc[-1] / 100.0

    short_trend = ema_12 > ema_26
    medium_trend = ema_26 > ema_50

    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal, confidence = "bullish", trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal, confidence = "bearish", trend_strength
    else:
        signal, confidence = "neutral", 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "ema12": float(ema_12.iloc[-1]),
            "ema26": float(ema_26.iloc[-1]),
            "ema50": float(ema_50.iloc[-1]),
            "trend_strength": float(trend_strength),
        },
    }


def calculate_mean_reversion_signals(prices_df: pd.DataFrame) -> dict:
    """
    Mean reversion strategy using statistical measures and Bollinger Bands
    """
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50

    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)

    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    price_vs_bb = (
        prices_df["close"].iloc[-1] - bb_lower.iloc[-1]
    ) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal, confidence = "bullish", min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal, confidence = "bearish", min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal, confidence = "neutral", 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": float(z_score.iloc[-1]),
            "price_vs_bb": float(price_vs_bb),
            "rsi_14": float(rsi_14.iloc[-1]),
            "rsi_28": float(rsi_28.iloc[-1]),
        },
    }


def calculate_momentum_signals(prices_df: pd.DataFrame, network_vol: float) -> dict:
    """
    Multi-factor momentum strategy with network volume confirmation
    """
    returns = prices_df["close"].pct_change()
    mom_1m = returns.rolling(21).sum().iloc[-1]
    mom_3m = returns.rolling(63).sum().iloc[-1]
    mom_6m = returns.rolling(126).sum().iloc[-1]

    volume_ma = prices_df["volume"].rolling(21).mean().iloc[-1]
    volume_confirmation = (
        prices_df["volume"].iloc[-1] > volume_ma
        and prices_df["volume"].iloc[-1] > network_vol
    )

    score_raw = 0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m
    if score_raw > 0 and volume_confirmation:
        signal, confidence = "bullish", min(score_raw * 5, 1.0)
    elif score_raw < 0 and volume_confirmation:
        signal, confidence = "bearish", min(-score_raw * 5, 1.0)
    else:
        signal, confidence = "neutral", 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1m": float(mom_1m),
            "momentum_3m": float(mom_3m),
            "momentum_6m": float(mom_6m),
            "volume_confirmation": volume_confirmation,
        },
    }


def calculate_volatility_signals(prices_df: pd.DataFrame) -> dict:
    """
    Volatility-based trading strategy
    """
    returns = prices_df["close"].pct_change()
    hist_vol = returns.rolling(21).std() * math.sqrt(252)

    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    atr = calculate_atr(prices_df)
    atr_ratio = atr / prices_df["close"]

    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if current_vol_regime < 0.8 and vol_z < -1:
        signal, confidence = "bullish", min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal, confidence = "bearish", min(abs(vol_z) / 3, 1.0)
    else:
        signal, confidence = "neutral", 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": float(hist_vol.iloc[-1]),
            "volatility_regime": float(current_vol_regime),
            "volatility_z_score": float(vol_z),
            "atr_ratio": float(atr_ratio.iloc[-1]),
        },
    }


def calculate_stat_arb_signals(prices_df: pd.DataFrame) -> dict:
    """
    Statistical arbitrage signals based on price action analysis
    """
    returns = prices_df["close"].pct_change()
    skew = returns.rolling(63).skew().iloc[-1]
    hurst = calculate_hurst_exponent(prices_df["close"])

    if hurst < 0.4 and skew > 1:
        signal, confidence = "bullish", (0.5 - hurst) * 2
    elif hurst < 0.4 and skew < -1:
        signal, confidence = "bearish", (0.5 - hurst) * 2
    else:
        signal, confidence = "neutral", 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": float(hurst),
            "skewness": float(skew),
        },
    }


def weighted_signal_combination(signals: dict, weights: dict) -> dict:
    """
    Combines multiple trading signals using a weighted approach
    """
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}
    weighted_sum = 0
    total_confidence = 0

    for strat, sig in signals.items():
        w = weights[strat]
        weighted_sum += signal_values[sig["signal"]] * sig["confidence"] * w
        total_confidence += sig["confidence"] * w

    final_score = weighted_sum / total_confidence if total_confidence > 0 else 0
    if final_score > 0.2:
        signal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {"signal": signal, "confidence": abs(final_score)}


def normalize_pandas(obj):
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    if isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize_pandas(i) for i in obj]
    return obj


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20):
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    return sma + 2 * std_dev, sma - 2 * std_dev


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]
    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

    df["+di"] = 100 * (df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["-di"] = 100 * (df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period).mean()

    return df[["adx", "+di", "-di"]]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    lags = range(2, max_lag)
    tau = [max(1e-8, np.sqrt(np.std(price_series[lag:] - price_series[:-lag]))) for lag in lags]
    try:
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        return m[0]
    except:
        return 0.5