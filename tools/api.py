from datetime import datetime, timedelta
import sys
from openai import OpenAI
import random
import pprint
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config2 import APCA_API_KEY_ID
alpaca_key = APCA_API_KEY_ID

from config2 import OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

import pandas as pd
import requests
import time  # if not already imported

from data.cache import get_cache
from data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
)

COINGECKO_HEADERS = {
    "accept": "application/json",
    "x-cg-demo-api-key": "CG-SohCK1ePBmQxyUdm29dSFWTA"
}

"""
#GEMINI PAIRS
COINGECKO_IDS = {
    "AAVE": "aave",
    "APE": "apecoin",
    "ATOM": "cosmos",
    "AVAX": "avalanche-2",
    "BCH": "bitcoin-cash",
    "BTC": "bitcoin",
    "COMP": "compound-governance-token",
    "CRV": "curve-dao-token",
    "DAI": "dai",
    "DOGE": "dogecoin",
    "DOT": "polkadot",
    "ETH": "ethereum",
    "FIL": "filecoin",
    "LINK": "chainlink",
    "LTC": "litecoin",
    "MANA": "decentraland",
    "MATIC": "matic-network",
    "MKR": "maker",
    "QNT": "quant-network",
    "SAND": "the-sandbox",
    "SHIB": "shiba-inu",
    "SOL": "solana",
    "SUSHI": "sushi",
    "UNI": "uniswap",
    "XRP": "ripple",
    "XTZ": "tezos",
    "YFI": "yearn-finance",
    "ZEC": "zcash",
}
"""
#ALPACA PAIRS
COINGECKO_IDS = {
    "AAVE": "aave",
    "AVAX": "avalanche-2",
    "BAT": "basic-attention-token",
    "BCH": "bitcoin-cash",
    "BTC": "bitcoin",
    "CRV": "curve-dao-token",
    "DOGE": "dogecoin",
    "DOT": "polkadot",
    "ETH": "ethereum",
    "GRT": "the-graph",
    "LINK": "chainlink",
    "LTC": "litecoin",
    "MKR": "maker",
    "SHIB": "shiba-inu",
    "SUSHI": "sushi",
    "UNI": "uniswap",
    "XTZ": "tezos",
    "YFI": "yearn-finance",
}

# Global cache instance
_cache = get_cache()

def fetch_with_retry(url, max_retries=20):
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=COINGECKO_HEADERS, timeout=10)
            if response.status_code == 200:
                import pprint
                pp = pprint.PrettyPrinter(indent=2)
                try:
                    data = response.json()
                    print("\n🌐 CoinGecko API response:")
                    #pp.pprint(data)
                    return data
                except Exception as e:
                    print(f"❌ Failed to parse JSON: {e}")
                    return None
            elif response.status_code == 429:
                if attempt == 1:
                    delay = random.uniform(80, 95)
                else:
                    base = random.uniform(10, 30)
                    delay = base * (2 ** (attempt - 2))
                print(f"⏳ CoinGecko rate limit hit. Retrying in {delay:.2f}s (attempt {attempt}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"❌ HTTP {response.status_code} for {url}: {response.text}")
                break
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            if attempt == 1:
                delay = random.uniform(80, 95)
            else:
                base = random.uniform(10, 30)
                delay = base * (2 ** (attempt - 2))
            print(f"⚠️ Request error (attempt {attempt}/{max_retries}) – {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)
    print("🛑 Max retries exceeded for CoinGecko.")
    return None

def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch historical daily close prices from Gemini API in year-sized chunks."""

    if cached := _cache.get_prices(ticker, start_date, end_date):
        print(f"📦 Loaded cached prices for {ticker} ({start_date} → {end_date}): {len(cached)} entries")
        return [Price(**item) for item in cached]

    import datetime
    import time

    symbol = ticker.replace("/", "").replace("-", "").lower()
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.datetime.strptime(end_date,   "%Y-%m-%d")
    url      = f"https://api.gemini.com/v2/candles/{symbol}/1day"

    all_candles = []
    period_start = start_dt
    while period_start <= end_dt:
        period_end = min(period_start + datetime.timedelta(days=365), end_dt)
        start_ts   = int(time.mktime(period_start.timetuple())) * 1000
        end_ts     = int(time.mktime(period_end.timetuple()))   * 1000

        print(f"🌐 Requesting Gemini candles for {symbol} {period_start.date()}→{period_end.date()}...")
        try:
            response = requests.get(url, params={"start": start_ts, "end": end_ts}, timeout=10)
        except requests.exceptions.ReadTimeout:
            print(f"⏳ Read timeout for {symbol} {period_start.date()}→{period_end.date()}, retrying in 2s…")
            time.sleep(2)
            response = requests.get(url, params={"start": start_ts, "end": end_ts}, timeout=10)
        if response.status_code != 200:
            raise Exception(f"❌ Gemini fetch error {response.status_code} for {ticker}: {response.text}")

        for ts, o, h, l, c, v in response.json():
            dt = datetime.datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d")
            all_candles.append(Price(time=dt, open=o, high=h, low=l, close=c, volume=int(v)))

        period_start = period_end + datetime.timedelta(days=1)

    print(f"📊 Parsed {len(all_candles)} candles from Gemini for {symbol}")
    _cache.set_prices(ticker, start_date, end_date, [p.model_dump() for p in all_candles])
    return all_candles

def get_financial_metrics(ticker: str, end_date: str) -> list[dict]:
    cache = get_cache()
    key = f"{ticker}::{end_date}"
    """Return crypto-adapted financial metrics for a given CoinGecko asset ID (e.g., 'bitcoin')."""
    # ─── Check cache ──────────────────────────────────────────────────────────────
    if cached := cache.get_financial_metrics(ticker, end_date):
        print(f"📦 Loaded cached financial metrics for {ticker}")
        return cached

    import requests

    # ─── Historical metrics ─────────────────────────────────────────────────────
    hist_metrics = get_historical_metrics(ticker, end_date)

    # Normalize symbol and lookup CoinGecko ID
    symbol = ticker.upper().replace("/USD", "").replace("-USD", "")
    asset_id = COINGECKO_IDS.get(symbol)
    if not asset_id:
        print(f"⚠️ No CoinGecko ID mapping for {ticker}")
        return []

    # ─── CoinGecko data ─────────────────────────────────────────────────────────
    cg_url = (
        f"https://api.coingecko.com/api/v3/coins/{asset_id}"
        "?localization=false&tickers=true&market_data=true"
        "&community_data=true&developer_data=true&sparkline=false"
    )
    cg = get_cached_coingecko_data(asset_id)
    if not cg:
        return []

    market_data = cg.get("market_data", {})
    data = [{
        "ticker": ticker,
        "report_period": "latest",
        "period": "ttm",
        "currency": "USD",
        "market_cap": market_data.get("market_cap", {}).get("usd"),
        "price_to_sales_ratio": market_data.get("market_cap", {}).get("usd") / market_data.get("total_volume", {}).get("usd", 1) if market_data.get("total_volume", {}).get("usd") else None,
        "current_price": market_data.get("current_price", {}).get("usd"),
        "ath": market_data.get("ath", {}).get("usd"),
        "volume_24h": market_data.get("total_volume", {}).get("usd"),
        "volume_to_market_cap": market_data.get("total_volume", {}).get("usd", 0) / market_data.get("market_cap", {}).get("usd", 1) if market_data.get("market_cap", {}).get("usd") else None,
        "price_change_pct_60d": market_data.get("price_change_percentage_60d_in_currency", {}).get("usd"),
        "price_change_pct_200d": market_data.get("price_change_percentage_200d_in_currency", {}).get("usd"),
        "price_change_pct_1y": market_data.get("price_change_percentage_1y_in_currency", {}).get("usd"),
    }]

    # ─── Core metadata ───────────────────────────────────────────────────────────
    coin_id           = cg.get("id")
    coin_symbol       = cg.get("symbol")
    coin_name         = cg.get("name")
    web_slug          = cg.get("web_slug")
    asset_platform_id = cg.get("asset_platform_id")
    platforms         = cg.get("platforms")
    detail_platforms  = cg.get("detail_platforms")
    hashing_algorithm = cg.get("hashing_algorithm")
    block_time_minutes= cg.get("block_time_in_minutes")
    categories        = cg.get("categories")
    country_origin    = cg.get("country_origin")
    genesis_date      = cg.get("genesis_date")

    # ─── Listing & notices ───────────────────────────────────────────────────────
    preview_listing    = cg.get("preview_listing")
    public_notice      = cg.get("public_notice")
    additional_notices = cg.get("additional_notices")
    localization_map   = cg.get("localization")

    # ─── Descriptions & links ─────────────────────────────────────────────────────
    desc = cg.get("description", {}) or {}
    description_en     = desc.get("en")
    description_de     = desc.get("de")
    links              = cg.get("links", {}) or {}
    twitter_handle     = links.get("twitter_screen_name")
    homepage_urls      = links.get("homepage")
    whitepaper_url     = links.get("whitepaper")
    blockchain_sites   = links.get("blockchain_site")
    forum_urls         = links.get("official_forum_url")
    chat_urls          = links.get("chat_url")
    announcement_urls  = links.get("announcement_url")
    snapshot_url       = links.get("snapshot_url")
    subreddit_url      = links.get("subreddit_url")
    repos = links.get("repos_url", {}) or {}
    repos_url_github   = repos.get("github")
    repos_url_bitbucket= repos.get("bitbucket")
    status_updates     = cg.get("status_updates")
    watchlist_users    = cg.get("watchlist_portfolio_users")
    market_cap_rank    = cg.get("market_cap_rank")
    sentiment_up_pct   = cg.get("sentiment_votes_up_percentage")
    sentiment_down_pct = cg.get("sentiment_votes_down_percentage")
    tickers_count      = len(cg.get("tickers", []))
    last_updated_global= cg.get("last_updated")

    # ─── Market data ─────────────────────────────────────────────────────────────
    md               = cg.get("market_data", {}) or {}
    market_cap       = md.get("market_cap", {}).get("usd")
    current_price    = md.get("current_price", {}).get("usd")
    volume_24h       = md.get("total_volume", {}).get("usd")
    circulating      = md.get("circulating_supply")
    total_supply     = md.get("total_supply")
    max_supply       = md.get("max_supply")
    fdv              = md.get("fully_diluted_valuation", {}).get("usd")

    # ─── Price change percentages ────────────────────────────────────────────────
    pct_24h  = md.get("price_change_percentage_24h")
    pct_7d   = md.get("price_change_percentage_7d")
    pct_14d  = md.get("price_change_percentage_14d")
    pct_30d  = hist_metrics.get("price_change_pct_30d", md.get("price_change_percentage_30d"))
    pct_60d  = md.get("price_change_percentage_60d")
    pct_200d = md.get("price_change_percentage_200d")
    pct_1y   = md.get("price_change_percentage_1y")

    # ─── ATH / ATL ────────────────────────────────────────────────────────────────
    ath      = md.get("ath", {}).get("usd")
    ath_date = md.get("ath_date", {}).get("usd")
    atl      = md.get("atl", {}).get("usd")
    atl_date = md.get("atl_date", {}).get("usd")

    # ─── Advanced scores ──────────────────────────────────────────────────────────
    coingecko_score       = cg.get("coingecko_score")
    liquidity_score       = cg.get("liquidity_score")
    public_interest_score = cg.get("public_interest_score")

    # ─── Public interest stats ────────────────────────────────────────────────────
    pis                  = cg.get("public_interest_stats", {}) or {}
    alexa_rank           = pis.get("alexa_rank")
    bing_matches         = pis.get("bing_matches")

    # ─── ROI ───────────────────────────────────────────────────────────────────────
    roi                  = cg.get("roi", {}) or {}
    roi_times            = roi.get("times")
    roi_currency         = roi.get("currency")
    roi_pct              = roi.get("percentage")

    # ─── Developer data ───────────────────────────────────────────────────────────
    dev                  = cg.get("developer_data", {}) or {}
    dev_forks            = dev.get("forks")
    dev_stars            = dev.get("stars")
    dev_subscribers      = dev.get("subscribers")
    dev_total_issues     = dev.get("total_issues")
    dev_closed_issues    = dev.get("closed_issues")
    dev_pr_merged        = dev.get("pull_requests_merged")
    pr_contributors      = dev.get("pull_request_contributors")
    changes_4w           = dev.get("code_additions_deletions_4_weeks", {}) or {}
    additions_4w         = changes_4w.get("additions")
    deletions_4w         = changes_4w.get("deletions")
    activity_series_4w   = dev.get("last_4_weeks_commit_activity_series", [])


    # ─── Community data ───────────────────────────────────────────────────────────
    comm                 = cg.get("community_data", {}) or {}
    facebook_likes       = comm.get("facebook_likes")
    twitter_followers    = comm.get("twitter_followers")
    reddit_subscribers   = comm.get("reddit_subscribers")
    reddit_posts_48h     = comm.get("reddit_average_posts_48h")
    reddit_comments_48h  = comm.get("reddit_average_comments_48h")
    reddit_active_48h    = comm.get("reddit_accounts_active_48h")
    telegram_user_count  = comm.get("telegram_channel_user_count")

    # ─── Primary ticker & conversions ─────────────────────────────────────────────
    primary = (cg.get("tickers") or [{}])[0] or {}
    primary_last          = primary.get("last")
    primary_volume        = primary.get("volume")
    primary_trust_score   = primary.get("trust_score")
    primary_spread_pct    = primary.get("bid_ask_spread_percentage")
    conv                  = primary.get("converted_last", {}) or {}
    conv_last_usd         = conv.get("usd")
    conv_last_btc         = conv.get("btc")
    conv_last_eth         = conv.get("eth")
    conv_vol              = primary.get("converted_volume", {}) or {}
    conv_vol_usd          = conv_vol.get("usd")
    conv_vol_btc          = conv_vol.get("btc")
    conv_vol_eth          = conv_vol.get("eth")

    # ─── Gemini pubticker ───────────────────────────────────────────────────────────
    gem_symbol = f"{symbol.lower()}usd"
    gem       = requests.get(f"https://api.gemini.com/v1/pubticker/{gem_symbol}").json()
    bid       = float(gem.get("bid", 0))
    ask       = float(gem.get("ask", 0))
    last      = float(gem.get("last", 0))
    mid       = float(gem.get("mid", (bid + ask) / 2))
    spread    = ask - bid

    # ─── Gemini order book & trades ────────────────────────────────────────────────
    book       = requests.get(
                    f"https://api.gemini.com/v1/book/{gem_symbol}?limit_bids=25&limit_asks=25"
                 ).json()
    top_bids   = book["bids"][:5]
    top_asks   = book["asks"][:5]
    trades     = requests.get(f"https://api.gemini.com/v1/trades/{gem_symbol}?limit_trades=500").json()
    trade_count    = len(trades)
    avg_trade_size = (sum(float(t["amount"]) for t in trades) / trade_count) if trade_count else None

    # ─── Gemini 1-day candle via v2 ────────────────────────────────────────────────
    v2_url = f"https://api.gemini.com/v2/candles/{gem_symbol}/1day"
    daily_open = daily_high = daily_low = daily_close = daily_volume = None
    try:
        resp = requests.get(v2_url)
        resp.raise_for_status()
        data = resp.json()
        if data and isinstance(data, list):
            _, daily_open, daily_high, daily_low, daily_close, daily_volume = max(data, key=lambda c: c[0])
    except Exception as e:
        print(f"⚠️ Failed to fetch Gemini v2 candle for {gem_symbol}: {e}")

    metrics = [{
        "ticker":                      ticker.upper(),
        "report_period":               "latest",
        "period":                      "ttm",
        "currency":                    "USD",
        # CoinGecko core metadata
        "coin_id":                     coin_id,
        "symbol":                      coin_symbol,
        "name":                        coin_name,
        "platforms":                   platforms,
        "detail_platforms":            detail_platforms,
        "hashing_algorithm":           hashing_algorithm,
        "block_time_minutes":          block_time_minutes,
        "categories":                  categories,
        "country_origin":              country_origin,
        "genesis_date":                genesis_date,
        "preview_listing":             preview_listing,
        "public_notice":               public_notice,
        "additional_notices":          additional_notices,
        "localization":                localization_map,
        "status_updates":              status_updates,
        "watchlist_portfolio_users":   watchlist_users,
        "market_cap_rank":             market_cap_rank,
        "sentiment_votes_up_pct":      sentiment_up_pct,
        "sentiment_votes_down_pct":    sentiment_down_pct,
        "tickers_count":               tickers_count,
        "last_updated_global":         last_updated_global,
        # Market data
        "market_cap":                  market_cap,
        "fully_diluted_valuation":     fdv or market_cap,
        "current_price":               current_price,
        "volume_24h":                  volume_24h,
        "circulating_supply":          circulating,
        "total_supply":                total_supply,
        "max_supply":                  max_supply,
        # Price change %
        "price_change_pct_24h":        pct_24h,
        "price_change_pct_7d":         pct_7d,
        "price_change_pct_14d":        pct_14d,
        "price_change_pct_30d":        pct_30d,
        "price_change_pct_60d":        pct_60d,
        "price_change_pct_200d":       pct_200d,
        "price_change_pct_1y":         pct_1y,
        # ATH/ATL
        "ath":                         ath,
        "ath_date":                    ath_date,
        "atl":                         atl,
        "atl_date":                    atl_date,
        # Advanced scores
        "coingecko_score":             coingecko_score,
        "liquidity_score":             liquidity_score,
        "public_interest_score":       public_interest_score,
        # Public interest stats
        "alexa_rank":                  alexa_rank,
        "bing_matches":                bing_matches,
        # ROI
        "roi_times":                   roi_times,
        "roi_currency":                roi_currency,
        "roi_percentage":              roi_pct,
        # Developer data
        "developer_forks":             dev_forks,
        "developer_stars":             dev_stars,
        "developer_subscribers":       dev_subscribers,
        "developer_total_issues":      dev_total_issues,
        "developer_closed_issues":     dev_closed_issues,
        "developer_pr_merged":         dev_pr_merged,
        "pr_contributors":             pr_contributors,
        "code_additions_4_weeks":      additions_4w,
        "code_deletions_4_weeks":      deletions_4w,
        "activity_series_4_weeks":     activity_series_4w,
        "developer_commit_count_4_weeks":     dev.get("commit_count_4_weeks"),
        "developer_code_additions_4_weeks":   additions_4w,
        "developer_code_deletions_4_weeks":   deletions_4w,
        "developer_commit_activity_series":   activity_series_4w,
        "developer_pr_contributors":          pr_contributors,
        "developer_activity": (
            (dev_stars or 0)
            + (dev_forks or 0)
            + (pr_contributors or 0)
            + (dev.get("commit_count_4_weeks") or 0)
        ),

        # Community data
        "facebook_likes":              facebook_likes,
        "twitter_followers":           twitter_followers,
        "reddit_subscribers":          reddit_subscribers,
        "reddit_posts_48h":            reddit_posts_48h,
        "reddit_comments_48h":         reddit_comments_48h,
        "reddit_accounts_active_48h":  reddit_active_48h,
        "telegram_channel_user_count": telegram_user_count,
        "active_addresses_24h": reddit_active_48h or volume_24h,
        # Primary ticker & conversions
        "primary_last":                primary_last,
        "primary_volume":              primary_volume,
        "primary_trust_score":         primary_trust_score,
        "primary_bid_ask_spread_pct":  primary_spread_pct,
        "converted_last_usd":          conv_last_usd,
        "converted_last_btc":          conv_last_btc,
        "converted_last_eth":          conv_last_eth,
        "converted_volume_usd":        conv_vol_usd,
        "converted_volume_btc":        conv_vol_btc,
        "converted_volume_eth":        conv_vol_eth,
        # Gemini pubticker
        "gemini_bid":                  bid,
        "gemini_ask":                  ask,
        "gemini_last":                 last,
        "gemini_mid_price":            mid,
        "gemini_spread":               spread,
        # Gemini book & trades
        "gemini_top_bids":             top_bids,
        "gemini_top_asks":             top_asks,
        "gemini_trade_count":          trade_count,
        "gemini_avg_trade_size":       avg_trade_size,
        # Gemini daily candle (v2)
        "gemini_daily_open":           daily_open,
        "gemini_daily_high":           daily_high,
        "gemini_daily_low":            daily_low,
        "gemini_daily_close":          daily_close,
        "gemini_daily_volume":         daily_volume,
        # Derived ratios (with historical override)
        "price_to_sales_ratio":            market_cap / volume_24h if market_cap and volume_24h else None,
        "enterprise_value_to_revenue_ratio": ((fdv or market_cap) / volume_24h) if volume_24h else None,
        "volume_to_market_cap": (volume_24h / market_cap) if market_cap and volume_24h else None,
        # Placeholders for non-applicable metrics
        "price_to_earnings_ratio":          "not applicable to crypto",
        "price_to_book_ratio":              "not applicable to crypto",
        "enterprise_value_to_ebitda_ratio": "not applicable to crypto",
        "free_cash_flow_yield":             "not applicable to crypto",
        "peg_ratio":                        "not applicable to crypto",
        "gross_margin":                     "not applicable to crypto",
        "operating_margin":                 "not applicable to crypto",
        "net_margin":                       "not applicable to crypto",
        "return_on_equity":                 "not applicable to crypto",
        "return_on_assets":                 "not applicable to crypto",
    }]

    _cache.set_financial_metrics(ticker, end_date, metrics)
    if not metrics:
        print(f"❌ No metrics found for {ticker}")
    else:
        print(f"✅ Metrics generated for {ticker}: {len(metrics[0].keys())} fields")

    _cache.set_financial_metrics(ticker, end_date, metrics)
    
    #pprint.pprint(metrics[0], sort_dicts=False)
    return metrics

def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch Gemini market data and dynamically map requested line_items."""
    try:
        # Normalize Gemini symbol format (e.g. BTC-USD → btcusd)
        gemini_symbol = ticker.lower().replace("-", "")

        # Fetch ticker data (price, volume)
        gemini_symbol = ticker.lower().replace("/", "").replace("-", "")
        ticker_url = f"https://api.gemini.com/v1/pubticker/{gemini_symbol}"

        print(f"🌐 Fetching Gemini pubticker from {ticker_url}...")
        try:
            ticker_resp = requests.get(ticker_url, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error for {gemini_symbol}: {e}")
            return []

        ticker_data = ticker_resp.json()
        last_price = float(ticker_data.get("last", 0))
        usd_volume = float(ticker_data.get("volume", {}).get("USD", 0))
        base_volume = float(ticker_data.get("volume", {}).get("BTC", 0))  # fallback to asset volume

        # Fetch symbol metadata
        details_url = f"https://api.gemini.com/v1/symbols/details/{gemini_symbol}"
        details_resp = requests.get(details_url)
        details = details_resp.json() if details_resp.status_code == 200 else {}

        # Create a dict to hold mapped results
        mapped = {
            "ticker": ticker.upper(),
            "report_period": end_date,
            "period": period,
            "currency": "USD"
        }

        for item in line_items:
            key = item.lower()

            if key == "revenue":
                mapped["revenue"] = usd_volume
            elif key == "gross_profit":
                mapped["gross_profit"] = usd_volume * 0.002  # mock 0.2% profit margin
            elif key == "price":
                mapped["price"] = last_price
            elif key == "volume":
                mapped["volume"] = base_volume
            elif key == "tick_size":
                mapped["tick_size"] = float(details.get("tick_size", 0))
            elif key == "min_order_size":
                mapped["min_order_size"] = float(details.get("min_order_size", 0))
            elif key == "quote_increment":
                mapped["quote_increment"] = float(details.get("quote_increment", 0))
            else:
                pass

        return [LineItem(**mapped)]

    except Exception as e:
        print(f"❌ Gemini fetch failed for {ticker}: {e}")
        return []

def get_historical_metrics(ticker: str, end_date: str) -> dict:
    if cached := _cache.get_historical_metrics(ticker, end_date):
        print(f"📦 Loaded cached historical metrics for {ticker}")
        return cached

    from time import sleep
    from datetime import datetime, timedelta

    start_30d = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

    max_retries = 20
    for attempt in range(1, max_retries + 1):
        prices = get_prices(ticker, start_30d, end_date)
        if prices:
            break
        wait = 2 ** attempt
        print(f"⏳ get_prices() failed for {ticker}. Retrying in {wait}s (attempt {attempt}/{max_retries})...")
        sleep(wait)
    else:
        print(f"❌ get_prices() failed for {ticker} after {max_retries} retries.")
        return {}

    df = prices_to_df(prices)
    pct_30d = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) if len(df) > 1 else 0.0
    volume_to_market_cap = df["volume"].mean() / (df["close"].mean() * 1_000_000)

    result = {
        "price_change_pct_30d": pct_30d,
        "volume_to_market_cap": volume_to_market_cap,
        "sentiment_votes_up_pct": 50.0  # placeholder
    }

    _cache.set_historical_metrics(ticker, end_date, result)
    return result

def get_insider_trades(ticker: str, end_date: str, start_date: str | None = None, limit: int = 10) -> list[InsiderTrade]:
    ticker = ticker.upper()
    if cached := _cache.get_insider_trades(ticker, end_date):
        print(f"📦 Loaded cached insider trades for {ticker}")
        return [InsiderTrade(**item) for item in cached]

    if ticker != "BTC/USD":
        print(f"ℹ️ Insider trades only available for BTC. Returning empty list for {ticker}")
        _cache.set_insider_trades(ticker, end_date, [])
        return []

    def get_with_backoff(url, params=None, headers=None, max_retries=1, timeout=1):
        delay = 2
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
                if response.status_code == 200:
                    return response
                elif response.status_code == 430 or "blacklisted" in response.text.lower():
                    print(f"⚠️ Blocked or throttled (attempt {attempt}) – waiting {delay}s before retry...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"❌ HTTP {response.status_code}: {response.text}")
                    return response
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                print(f"⏳ Timeout or connection error on attempt {attempt} – waiting {delay}s: {e}")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                return None
        print("🛑 Max retries exceeded.")
        return None

    url = f"https://api.blockchair.com/bitcoin/transactions?limit={limit}"
    response = get_with_backoff(url)
    if not response or response.status_code != 200:
        print("❌ Failed to fetch Blockchair data.")
        return []

    txs = response.json().get("data", [])
    trades = []
    for tx in txs:
        usd = tx.get("input_total_usd", 0)
        btc = tx.get("input_total", 0) / 1e8
        if usd >= 1_000_000:
            trades.append({
                "insider_name": "Unknown Wallet",
                "issuer": ticker,
                "name": "Whale",
                "title": "N/A",
                "is_board_director": False,
                "transaction_shares": btc,
                "transaction_price_per_share": usd / btc if btc else 0,
                "transaction_value": usd,
                "shares_owned_before_transaction": None,
                "shares_owned_after_transaction": None,
                "security_title": "Token Transfer",
                "filing_date": tx["date"]
            })

    _cache.set_insider_trades(ticker, end_date, trades)
    return trades

_in_progress = {}

def get_cached_coingecko_data(asset_id: str) -> dict | None:
    key = f"coingecko:{asset_id}"
    if cached := _cache.get_cached_data(key):
        print(f"📦 Loaded cached CoinGecko data for {asset_id}")
        return cached

    # prevent duplicate simultaneous fetches
    if asset_id in _in_progress:
        while asset_id in _in_progress:
            time.sleep(0.25)
        return _cache.get_cached_data(key)  # re-check cache after waiting

    _in_progress[asset_id] = True
    try:
        url = (
            f"https://api.coingecko.com/api/v3/coins/{asset_id}"
            "?localization=false&tickers=true&market_data=true"
            "&community_data=true&developer_data=true&sparkline=false"
        )
        data = fetch_with_retry(url)
        if data:
            _cache.set_cached_data(key, data)
        return data
    finally:
        del _in_progress[asset_id]

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, APITimeoutError

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(APITimeoutError),
)
def _safe_classify_sentiment(client, prompt):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

def classify_sentiment(text: str) -> str:
    """Use OpenAI Chat API to classify sentiment as 'positive', 'neutral', or 'negative'."""
    from config2 import OPENAI_API_KEY

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = (
        "Classify the sentiment of the following news text as strictly one of: "
        "'positive', 'neutral', or 'negative'.\n\n"
        f"Text:\n{text.strip()}"
    )

    try:
        response = _safe_classify_sentiment(client, prompt)
        sentiment = response.choices[0].message.content.strip().lower()
        return sentiment if sentiment in {"positive", "neutral", "negative"} else "neutral"
    except Exception as e:
        print(f"⚠️ classify_sentiment failed: {e}")
        return "neutral"

def get_company_news(ticker: str, end_date: str, start_date: str | None = None, limit: int = 10) -> list[CompanyNews]:
    """Fetch company news using Alpaca API with cache and per-ticker locking."""
    from config2 import APCA_API_KEY_ID, APCA_API_SECRET_KEY
    from data.cache import get_cache, get_lock

    cache = get_cache()
    symbol = ticker.replace("/", "").replace("-", "").upper()

    lock = get_lock(symbol)  # <- must lock on the normalized symbol, not raw ticker

    with lock:
        # Check cache first
        if cached_data := cache.get_company_news(symbol):
            filtered_data = [
                CompanyNews(**news)
                for news in cached_data
                if (start_date is None or news["date"] >= start_date)
                and news["date"] <= end_date
            ]
            filtered_data.sort(key=lambda x: x.date, reverse=True)
            if filtered_data:
                print(f"📦 Loaded cached news for {symbol}: {len(filtered_data)} articles")
                return filtered_data

        # If not cached, fetch
        print(f"📡 Requesting Alpaca news for {symbol}...")

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": APCA_API_KEY_ID,
            "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY,
        }

        base_url = "https://data.alpaca.markets/v1beta1/news"
        params = {
            "symbols": symbol,
            "limit": limit,
            "sort": "desc",
            "include_content": "true",
            "exclude_contentless": "true",
        }

        if start_date and end_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=1)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            params["start"] = start_dt.strftime("%Y-%m-%d")
            params["end"] = end_dt.strftime("%Y-%m-%d")

        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching Alpaca news: {response.status_code} - {response.text}")

        data = response.json()
        articles = data.get("news", [])
        print(f"📄 Retrieved {len(articles)} news articles from Alpaca")

        news_items = []
        for a in articles:
            news_items.append(CompanyNews(
                ticker=symbol,
                title=a.get("headline"),
                author=a.get("source"),
                source=a.get("source"),
                date=a.get("created_at")[:10],
                url=a.get("url"),
                sentiment=None
            ))

        for item in news_items:
            item.sentiment = classify_sentiment(item.title)

        cache.set_company_news(symbol, [n.model_dump() for n in news_items])
        print(f"✅ Done fetching Alpaca news for {symbol} — {len(news_items)} articles")

        return news_items


def get_market_cap(ticker: str, end_date: str) -> float | None:
    if cached := _cache.get_market_cap(ticker, end_date):
        print(f"📦 Loaded cached market cap for {ticker}")
        return cached

    base = ticker.upper().replace("/USD", "").replace("-USD", "")
    coingecko_id = COINGECKO_IDS.get(base)

    if not coingecko_id:
        print(f"⚠️ No CoinGecko ID mapping for {ticker}")
        return None

    url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}"
    cg = get_cached_coingecko_data(coingecko_id)
    if not cg:
        return None

    market_cap = cg.get("market_data", {}).get("market_cap", {}).get("usd")
    if market_cap:
        _cache.set_market_cap(ticker, end_date, float(market_cap))
    return float(market_cap) if market_cap else None


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)

# --- TEST BLOCK (comment out after use) ---
"""
ticker = "BTC/USD"
start_date = "2025-06-12"
end_date = "2025-06-12"

MAX_ITEMS = 3  # limit how many results to show

print("\n=== get_prices ===")
prices = get_prices(ticker, start_date, end_date)
print([p.model_dump() for p in prices[:MAX_ITEMS]])

print("\n=== get_financial_metrics ===")
metrics = get_financial_metrics(ticker)
print(metrics[:MAX_ITEMS])

print("\n=== search_line_items ===")
line_items = search_line_items(ticker, ["revenue", "gross_profit"], end_date)
print([li.model_dump() for li in line_items[:MAX_ITEMS]])

print("\n=== get_insider_trades ===")
trades = get_insider_trades(ticker, end_date, start_date)
print([t.model_dump() for t in trades[:MAX_ITEMS]])

print("\n=== get_company_news ===")
news = get_company_news(ticker, end_date, start_date)
print([n.model_dump() for n in news[:MAX_ITEMS]])

print("\n=== get_market_cap ===")
print(get_market_cap(ticker, end_date))
"""