class Cache:
    """In-memory cache for API responses, keyed by ticker and date where needed."""

    def __init__(self):
        # Composite-key caches
        self._prices_cache: dict[str, list[dict[str, any]]] = {}
        self._financial_metrics_cache: dict[str, list[dict[str, any]]] = {}
        self._historical_metrics_cache: dict[str, dict[str, any]] = {}
        self._insider_trades_cache: dict[str, list[dict[str, any]]] = {}
        self._market_cap_cache: dict[str, float] = {}
        # Single-key caches
        self._line_items_cache: dict[str, list[dict[str, any]]] = {}
        self._company_news_cache: dict[str, list[dict[str, any]]] = {}

    def _merge_data(self, existing: list[dict] | None, new_data: list[dict], key_field: str) -> list[dict]:
        """Merge existing and new data, avoiding duplicates based on a key field."""
        if not existing:
            return new_data
        existing_keys = {item[key_field] for item in existing}
        merged = existing.copy()
        merged.extend([item for item in new_data if item[key_field] not in existing_keys])
        return merged

    # ─── Prices ────────────────────────────────────────────────────────────────
    def get_prices(self, ticker: str, start: str, end: str) -> list[dict[str, any]] | None:
        return self._prices_cache.get(f"{ticker}::{start}::{end}")

    def set_prices(self, ticker: str, start: str, end: str, data: list[dict[str, any]]):
        key = f"{ticker}::{start}::{end}"
        self._prices_cache[key] = self._merge_data(self._prices_cache.get(key), data, key_field="time")

    # ─── Financial Metrics ─────────────────────────────────────────────────────
    def get_financial_metrics(self, ticker: str, end: str) -> list[dict[str, any]] | None:
        return self._financial_metrics_cache.get(f"{ticker}::{end}")

    def set_financial_metrics(self, ticker: str, end: str, data: list[dict[str, any]]):
        key = f"{ticker}::{end}"
        self._financial_metrics_cache[key] = self._merge_data(self._financial_metrics_cache.get(key), data, key_field="report_period")

    # ─── Historical Metrics ────────────────────────────────────────────────────
    def get_historical_metrics(self, ticker: str, end: str) -> dict[str, any] | None:
        return self._historical_metrics_cache.get(f"{ticker}::{end}")

    def set_historical_metrics(self, ticker: str, end: str, data: dict[str, any]):
        self._historical_metrics_cache[f"{ticker}::{end}"] = data

    # ─── Insider Trades ─────────────────────────────────────────────────────────
    def get_insider_trades(self, ticker: str, end: str) -> list[dict[str, any]] | None:
        return self._insider_trades_cache.get(f"{ticker}::{end}")

    def set_insider_trades(self, ticker: str, end: str, data: list[dict[str, any]]):
        key = f"{ticker}::{end}"
        self._insider_trades_cache[key] = self._merge_data(self._insider_trades_cache.get(key), data, key_field="filing_date")

    # ─── Market Cap ─────────────────────────────────────────────────────────────
    def get_market_cap(self, ticker: str, end: str) -> float | None:
        return self._market_cap_cache.get(f"{ticker}::{end}")

    def set_market_cap(self, ticker: str, end: str, value: float):
        self._market_cap_cache[f"{ticker}::{end}"] = value

    # ─── Line Items ─────────────────────────────────────────────────────────────
    def get_line_items(self, ticker: str) -> list[dict[str, any]] | None:
        return self._line_items_cache.get(ticker)

    def set_line_items(self, ticker: str, data: list[dict[str, any]]):
        self._line_items_cache[ticker] = self._merge_data(self._line_items_cache.get(ticker), data, key_field="report_period")

    # ─── Company News ───────────────────────────────────────────────────────────
    def get_company_news(self, ticker: str) -> list[dict[str, any]] | None:
        return self._company_news_cache.get(ticker)

    def set_company_news(self, ticker: str, data: list[dict[str, any]]):
        self._company_news_cache[ticker] = self._merge_data(self._company_news_cache.get(ticker), data, key_field="date")


# Global cache instance
_cache = Cache()

def get_cache() -> Cache:
    """Get the global cache instance."""
    return _cache