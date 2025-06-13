
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

# Global cache instance
_cache = get_cache()


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or API."""
    # Check cache first
    if cached_data := _cache.get_prices(ticker):
        # Filter cached data by date range and convert to Price objects
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    # If not in cache or no data in range, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    max_retries = 10
    base_wait = 5  # starting delay in seconds
    
    retry_attempt = 1
    while retry_attempt <= max_retries:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                break
            elif response.status_code == 429:
                retry_after_header = response.headers.get("Retry-After")
                wait = int(retry_after_header) if retry_after_header else base_wait * (2 ** (retry_attempt - 1))
                print(f"⚠️ Rate limit hit for {ticker} (Prices). Retrying in {wait}s... (Attempt {retry_attempt}/{max_retries})")
                time.sleep(wait)
                retry_attempt += 1
            else:
                raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            wait = base_wait * (2 ** (retry_attempt - 1))
            print(f"🌐 Connection issue for {ticker} (Prices). Attempt {retry_attempt}/{max_retries}: {e}. Retrying in {wait}s...")
            time.sleep(wait)
            retry_attempt += 1
    else:
        raise Exception(f"❌ Max retries exceeded for {ticker} (Prices). Last status: {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    price_response = PriceResponse(**response.json())
    prices = price_response.prices

    if not prices:
        return []

    # Cache the results as dicts
    _cache.set_prices(ticker, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # Check cache first
    if cached_data := _cache.get_financial_metrics(ticker):
        # Filter cached data by date and limit
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        if filtered_data:
            return filtered_data[:limit]

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    max_retries = 10
    base_wait = 5
    retry_attempt = 1
    
    while retry_attempt <= max_retries:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                break
            elif response.status_code == 429:
                retry_after_header = response.headers.get("Retry-After")
                wait = int(retry_after_header) if retry_after_header else base_wait * (2 ** (retry_attempt - 1))
                print(f"⚠️ Rate limit hit for {ticker} (Metrics). Retrying in {wait}s... (Attempt {retry_attempt}/{max_retries})")
                time.sleep(wait)
                retry_attempt += 1
            else:
                raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            wait = base_wait * (2 ** (retry_attempt - 1))
            print(f"🌐 Connection issue for {ticker} (Metrics). Attempt {retry_attempt}/{max_retries}: {e}. Retrying in {wait}s...")
            time.sleep(wait)
            retry_attempt += 1
    else:
        raise Exception(f"❌ Max retries exceeded for {ticker} (Metrics). Last status: {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    metrics_response = FinancialMetricsResponse(**response.json())
    # Return the FinancialMetrics objects directly instead of converting to dict
    financial_metrics = metrics_response.financial_metrics

    if not financial_metrics:
        return []

    # Cache the results as dicts
    _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch line items from API."""
    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }


    max_retries = 10
    base_delay = 5  # seconds

    retry_attempt = 1
    while retry_attempt <= max_retries:
        try:
            response = requests.post(url, headers=headers, json=body, timeout=10)
            if response.status_code == 200:
                break
            elif response.status_code == 429:
                retry_after_header = response.headers.get("Retry-After")
                wait = int(retry_after_header) if retry_after_header else base_delay * (2 ** (retry_attempt - 1))
                print(f"⚠️ Rate limit hit for {ticker} (Line Items). Retrying in {wait}s... (Attempt {retry_attempt}/{max_retries})")
                time.sleep(wait)
                retry_attempt += 1
            else:
                raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            wait = base_delay * (2 ** (retry_attempt - 1))
            print(f"🌐 Connection issue for {ticker} (Line Items). Attempt {retry_attempt}/{max_retries}: {e}. Retrying in {wait}s...")
            time.sleep(wait)
            retry_attempt += 1
    else:
        raise Exception(f"❌ Max retries exceeded for {ticker} (Line Items). Last status: {response.status_code} - {response.text}")
    data = response.json()
    response_model = LineItemResponse(**data)
    search_results = response_model.search_results
    if not search_results:
        return []

    # Cache the results
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Check cache first
    if cached_data := _cache.get_insider_trades(ticker):
        # Filter cached data by date range
        filtered_data = [InsiderTrade(**trade) for trade in cached_data 
                        if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                        and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_trades = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"
        
        max_retries = 10
        base_wait = 5
        retry_attempt = 1
        
        while retry_attempt <= max_retries:
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    break
                elif response.status_code == 429:
                    retry_after_header = response.headers.get("Retry-After")
                    wait = int(retry_after_header) if retry_after_header else base_wait * (2 ** (retry_attempt - 1))
                    print(f"⚠️ Rate limit hit for {ticker} (Insider Trades). Retrying in {wait}s... (Attempt {retry_attempt}/{max_retries})")
                    time.sleep(wait)
                    retry_attempt += 1
                else:
                    raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                wait = base_wait * (2 ** (retry_attempt - 1))
                print(f"🌐 Connection issue for {ticker} (Insider Trades). Attempt {retry_attempt}/{max_retries}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                retry_attempt += 1
        else:
            raise Exception(f"❌ Max retries exceeded for {ticker} (Insider Trades). Last status: {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades
        
        if not insider_trades:
            break
            
        all_trades.extend(insider_trades)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break
            
        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results
    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Check cache first
    if cached_data := _cache.get_company_news(ticker):
        # Filter cached data by date range
        filtered_data = [CompanyNews(**news) for news in cached_data 
                         if (start_date is None or news["date"] >= start_date)
                         and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            print(f"📦 Loaded cached news for {ticker}: {len(filtered_data)} articles")
            return filtered_data

    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date

    max_retries = 10
    base_wait = 5  # seconds
    retry_attempt = 1

    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"

        print(f"🔁 Fetching news page for {ticker} ending at {current_end_date}")

        while retry_attempt <= max_retries:
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    break
                elif response.status_code == 429:
                    retry_after_header = response.headers.get("Retry-After")
                    wait = int(retry_after_header) if retry_after_header else base_wait * (2 ** (retry_attempt - 1))
                    print(f"⚠️ Rate limit hit for {ticker} (News). Retrying in {wait}s... (Attempt {retry_attempt}/{max_retries})")
                    time.sleep(wait)
                    retry_attempt += 1
                else:
                    raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                wait = base_wait * (2 ** (retry_attempt - 1))
                print(f"🌐 Connection issue for {ticker} (Attempt {retry_attempt}/{max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
                retry_attempt += 1
        else:
            raise Exception(f"❌ Max retries exceeded for {ticker} (News). Last status: {response.status_code} - {response.text}")

        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news

        print(f"📄 Retrieved {len(company_news)} news items for {ticker} in this page")

        if not company_news:
            break

        all_news.extend(company_news)

        if not start_date or len(company_news) < limit:
            break

        new_end_date = min(news.date for news in company_news).split('T')[0]
        if new_end_date == current_end_date:
            print(f"🔁 Pagination stalled for {ticker} at {current_end_date} — stopping to prevent infinite loop.")
            break
        current_end_date = new_end_date
        if current_end_date <= start_date:
            break

    if not all_news:
        print(f"⚠️ No news found for {ticker}. Exiting.")
    else:
        print(f"✅ Done fetching news for {ticker} — {len(all_news)} articles total")

    _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    return all_news



def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap from the API."""
    financial_metrics = get_financial_metrics(ticker, end_date)
    market_cap = financial_metrics[0].market_cap
    if not market_cap:
        return None

    return market_cap


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

ticker = "AAPL"
start_date = "2025-06-12"
end_date = "2025-06-12"

MAX_ITEMS = 3  # limit how many results to show

print("\n=== get_prices ===")
prices = get_prices(ticker, start_date, end_date)
print([p.model_dump() for p in prices[:MAX_ITEMS]])

print("\n=== get_financial_metrics ===")
metrics = get_financial_metrics(ticker, end_date)
print([m.model_dump() for m in metrics[:MAX_ITEMS]])

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