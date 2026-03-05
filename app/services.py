# app/services.py
"""
Services layer: API clients, ingestion and helpers.

Design goals:
- Provide simple async-friendly functions used by core:
    * fetch_price_history(cfg, symbol) -> pd.DataFrame | None
    * fetch_fundamentals(cfg, symbol) -> dict
    * fetch_all_rss(cfg) -> list[dict]
    * fetch_rss(cfg, url, since=None) -> list[dict]
    * fetch_all_telegram(cfg) -> list[dict]
    * fetch_recent_messages_for_symbol(cfg, symbol, since=None) -> list[dict]
    * fetch_sec_filings(cfg, symbol) -> list[dict]
- Use preferred providers in order with fallbacks
- Respect API rate limits and provide simple caching
- Works both sync and async: if aiohttp available, uses async http; else runs blocking requests in threadpool
- Logs thoroughly and fails gracefully
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger("app.services")
logger.addHandler(logging.NullHandler())

# Optional libraries (import if present; otherwise graceful fallback)
try:
    import aiohttp  # type: ignore

    _HAS_AIOHTTP = True
except Exception:
    aiohttp = None  # type: ignore
    _HAS_AIOHTTP = False

try:
    import requests  # type: ignore

    _HAS_REQUESTS = True
except Exception:
    requests = None  # type: ignore
    _HAS_REQUESTS = False

try:
    import yfinance as yf  # type: ignore

    _HAS_YFINANCE = True
except Exception:
    yf = None  # type: ignore
    _HAS_YFINANCE = False

try:
    from alpha_vantage.timeseries import TimeSeries  # type: ignore

    _HAS_ALPHA = True
except Exception:
    TimeSeries = None  # type: ignore
    _HAS_ALPHA = False

try:
    from pycoingecko import CoinGeckoAPI  # type: ignore

    _HAS_COINGECKO = True
except Exception:
    CoinGeckoAPI = None  # type: ignore
    _HAS_COINGECKO = False

try:
    import feedparser  # type: ignore

    _HAS_FEEDPARSER = True
except Exception:
    feedparser = None  # type: ignore
    _HAS_FEEDPARSER = False

# Telegram clients optional
try:
    from telethon import TelegramClient  # type: ignore

    _HAS_TELETHON = True
except Exception:
    TelegramClient = None  # type: ignore
    _HAS_TELETHON = False

# Local DB module (for caching writes)
import app.db as db  # expected to exist from earlier steps

# ThreadPool for blocking calls
_THREAD_POOL: Optional[ThreadPoolExecutor] = None
_THREAD_POOL_LOCK = threading.Lock()


def _get_threadpool(max_workers: int = 8) -> ThreadPoolExecutor:
    global _THREAD_POOL
    with _THREAD_POOL_LOCK:
        if _THREAD_POOL is None:
            _THREAD_POOL = ThreadPoolExecutor(max_workers=max_workers)
        return _THREAD_POOL


# Simple persistent file cache for service responses (useful across restarts)
class FileCache:
    """
    Simple JSON-file cache with TTL. Key must be string-safe.
    """

    def __init__(self, cache_dir: str = "./data/cache", default_ttl_sec: int = 3600 * 6):
        self.cache_dir = cache_dir
        self.ttl = default_ttl_sec
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = key.replace("/", "_").replace(":", "_").replace(" ", "_")
        return os.path.join(self.cache_dir, f"{safe}.json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            p = self._path(key)
            if not os.path.exists(p):
                return None
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            ts = datetime.fromisoformat(obj.get("_cached_at"))
            if (datetime.utcnow() - ts).total_seconds() > obj.get("_ttl", self.ttl):
                try:
                    os.remove(p)
                except Exception:
                    pass
                return None
            return obj.get("value")
        except Exception:
            logger.exception("FileCache.get failed for %s", key)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            p = self._path(key)
            payload = {"_cached_at": datetime.utcnow().isoformat(), "_ttl": ttl or self.ttl, "value": value}
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            logger.exception("FileCache.set failed for %s", key)


# In-memory TTL cache (lightweight)
class MemCache:
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            rec = self._store.get(key)
            if not rec:
                return None
            if rec["expiry"] < time.time():
                del self._store[key]
                return None
            return rec["value"]

    def set(self, key: str, value: Any, ttl: int = 3600):
        with self._lock:
            self._store[key] = {"value": value, "expiry": time.time() + ttl}


_file_cache = FileCache()
_mem_cache = MemCache()

# Rate-limiter state (naive token-bucket per provider)
_RATE_STATE: Dict[str, Dict[str, Any]] = {}
_RATE_LOCK = threading.Lock()


def _allow_rate(provider: str, calls_per_min: int = 5) -> bool:
    """
    Naive per-provider rate-limiter: allow calls_per_min per minute.
    """
    with _RATE_LOCK:
        state = _RATE_STATE.setdefault(provider, {"count": 0, "reset": time.time() + 60})
        now = time.time()
        if now >= state["reset"]:
            state["count"] = 0
            state["reset"] = now + 60
        if state["count"] >= calls_per_min:
            return False
        state["count"] += 1
        return True


# Global service clients (initialized by initialize)
_SERVICES = {
    "alpha_ts": None,
    "cg": None,
    "telethon_client": None,
    "aiohttp_session": None,
}


# -----------------------
# Initialization / shutdown
# -----------------------
def initialize(cfg: Any) -> None:
    """
    Initialize service clients and resources.

    - Setup AlphaVantage client (if key)
    - Setup CoinGecko client
    - Prepare Telethon client if TELEGRAM API keys present
    - Create aiohttp session if available
    """
    logger.info("services.initialize starting")
    # alpha vantage
    av_key = getattr(cfg, "ALPHAVANTAGE_API_KEY", None)
    if av_key and _HAS_ALPHA:
        try:
            _SERVICES["alpha_ts"] = TimeSeries(key=av_key, output_format="pandas")
            logger.info("AlphaVantage client ready")
        except Exception:
            logger.exception("Failed to init AlphaVantage client")
            _SERVICES["alpha_ts"] = None
    else:
        _SERVICES["alpha_ts"] = None

    # coingecko
    if _HAS_COINGECKO:
        try:
            _SERVICES["cg"] = CoinGeckoAPI()
            logger.info("CoinGecko client ready")
        except Exception:
            logger.exception("Failed to init CoinGecko client")
            _SERVICES["cg"] = None

    # Telethon / Telegram
    tg_api_id = getattr(cfg, "TELEGRAM_API_ID", None)
    tg_api_hash = getattr(cfg, "TELEGRAM_API_HASH", None)
    if tg_api_id and tg_api_hash and _HAS_TELETHON:
        try:
            # session name from cfg or default
            session_name = getattr(cfg, "TELETHON_SESSION", "services_telethon_session")
            client = TelegramClient(session_name, tg_api_id, tg_api_hash)
            # do not start here; caller will start if needed
            _SERVICES["telethon_client"] = client
            logger.info("Telethon client prepared (not started)")
        except Exception:
            logger.exception("Failed to prepare Telethon client")
            _SERVICES["telethon_client"] = None
    else:
        _SERVICES["telethon_client"] = None

    # aiohttp
    if _HAS_AIOHTTP:
        try:
            # share a single session
            loop = asyncio.get_event_loop()
            # create session in loop
            async def _create_session():
                return aiohttp.ClientSession()
            sess = loop.run_until_complete(_create_session())
            _SERVICES["aiohttp_session"] = sess
            logger.info("aiohttp session created")
        except Exception:
            logger.exception("Failed to create aiohttp session")
            _SERVICES["aiohttp_session"] = None
    logger.info("services.initialize done")


def shutdown(cfg: Any = None) -> None:
    """
    Shutdown clients (aiohttp session, telethon client) and threadpool if created.
    """
    logger.info("services.shutdown starting")
    if _SERVICES.get("aiohttp_session") is not None:
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_SERVICES["aiohttp_session"].close())
            _SERVICES["aiohttp_session"] = None
        except Exception:
            logger.exception("Failed to close aiohttp session")
    if _SERVICES.get("telethon_client") is not None:
        try:
            client = _SERVICES["telethon_client"]
            if client and getattr(client, "is_connected", False):
                loop = asyncio.get_event_loop()
                loop.run_until_complete(client.disconnect())
            _SERVICES["telethon_client"] = None
        except Exception:
            logger.exception("Failed to shutdown Telethon client")
    # threadpool
    global _THREAD_POOL
    if _THREAD_POOL is not None:
        try:
            _THREAD_POOL.shutdown(wait=True)
            _THREAD_POOL = None
        except Exception:
            logger.exception("Failed to shutdown threadpool")
    logger.info("services.shutdown done")


# -----------------------
# HTTP helpers (async or sync)
# -----------------------
async def _http_get_async(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """
    Perform GET request using aiohttp and return JSON/text wrapper.
    """
    if not _HAS_AIOHTTP:
        return await asyncio.get_event_loop().run_in_executor(_get_threadpool(), lambda: _http_get_sync(url, params, headers, timeout))
    session = _SERVICES.get("aiohttp_session")
    if session is None:
        # create temporary session
        async with aiohttp.ClientSession() as s:
            try:
                async with s.get(url, params=params, headers=headers, timeout=timeout) as resp:
                    text = await resp.text()
                    try:
                        return {"status": resp.status, "text": text, "json": await resp.json()}
                    except Exception:
                        return {"status": resp.status, "text": text}
            except Exception:
                logger.exception("aiohttp GET failed for %s", url)
                return None
    try:
        async with session.get(url, params=params, headers=headers, timeout=timeout) as resp:
            text = await resp.text()
            try:
                js = await resp.json()
            except Exception:
                js = None
            return {"status": resp.status, "text": text, "json": js}
    except Exception:
        logger.exception("aiohttp GET failed for %s", url)
        return None


def _http_get_sync(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """
    Synchronous GET using requests.
    """
    if not _HAS_REQUESTS:
        logger.error("requests library not installed; cannot perform HTTP GET for %s", url)
        return None
    try:
        resp = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
        text = resp.text
        try:
            js = resp.json()
        except Exception:
            js = None
        return {"status": resp.status_code, "text": text, "json": js}
    except Exception:
        logger.exception("requests GET failed for %s", url)
        return None


# -----------------------
# Price & fundamentals fetching
# -----------------------
async def fetch_price_history(cfg: Any, symbol: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch price history for a symbol.

    Logic:
    - If symbol looks like crypto (contains - or lowercase or config says), route to CoinGecko
    - Prefer yfinance (easy & rich) -> returns pd.DataFrame with ['timestamp','open','high','low','close','adj_close','volume']
    - If fails, try AlphaVantage (rate-limited)
    - Cache responses in file-cache and optionally persist to DB via db.save_price_history

    Returns pandas DataFrame or None.
    """
    if not symbol:
        return None
    sym = symbol.upper()
    cache_key = f"price:{sym}:{period}:{interval}"
    # try mem cache
    cached = _mem_cache.get(cache_key)
    if cached is not None:
        return pd.read_json(cached)

    # try file cache
    fc = _file_cache.get(cache_key)
    if fc is not None:
        try:
            df = pd.read_json(json.dumps(fc))
            _mem_cache.set(cache_key, df.to_json(), ttl=3600)
            return df
        except Exception:
            pass

    # detect crypto heuristics: user may mark crypto by all-lowercase or prefix 'CRYPTO:'
    is_crypto = False
    # common pattern: BTC, ETH are cryptos but uppercase... use cfg CRYPTO_LIST override or symbol in known list
    crypto_list = getattr(cfg, "KNOWN_CRYPTO_SYMBOLS", ["BTC", "ETH", "USDT", "XRP", "LTC"])
    if sym in crypto_list or getattr(cfg, "FORCE_CRYPTO", False) or getattr(cfg, "CRYPTO_SUFFIX", None) and sym.endswith(getattr(cfg, "CRYPTO_SUFFIX")):
        is_crypto = True

    # 1) yfinance path
    if not is_crypto and _HAS_YFINANCE:
        try:
            # yfinance has blocking IO; run in threadpool for async compatibility
            def _yf_fetch():
                try:
                    ticker = yf.Ticker(sym)
                    hist = ticker.history(period=period, interval=interval, actions=False)
                    if hist is None or hist.empty:
                        return None
                    hist = hist.reset_index()
                    # unify columns
                    hist = hist.rename(columns={"Date": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", "Adj Close": "adj_close"})
                    if "adj_close" not in hist.columns and "Close" in hist.columns:
                        hist["adj_close"] = hist["close"]
                    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
                    return hist[["timestamp", "open", "high", "low", "close", "adj_close", "volume"]]
                except Exception:
                    logger.exception("yfinance fetch failed for %s", sym)
                    return None

            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(_get_threadpool(), _yf_fetch)
            if df is not None and not df.empty:
                # cache and persist
                _mem_cache.set(cache_key, df.to_json(), ttl=3600)
                _file_cache.set(cache_key, json.loads(df.to_json()), ttl=getattr(cfg, "PRICE_CACHE_TTL", 3600))
                try:
                    # persist to DB asynchronously via threadpool - not awaited here
                    loop.run_in_executor(_get_threadpool(), lambda: db.save_price_history(cfg, sym, df))
                except Exception:
                    logger.exception("Failed to persist price history to DB for %s", sym)
                return df
        except Exception:
            logger.exception("yfinance path failed for %s", sym)

    # 2) AlphaVantage fallback (TIME_SERIES_DAILY_ADJUSTED)
    av_client = _SERVICES.get("alpha_ts")
    if av_client is not None:
        try:
            if not _allow_rate("alpha", getattr(cfg, "ALPHAVANTAGE_RATE_PER_MIN", 5)):
                logger.debug("AlphaVantage rate limit reached, skipping for now")
            else:
                # TimeSeries.get_daily_adjusted returns (DataFrame, meta) for pandas output_format
                def _av_fetch():
                    try:
                        data, meta = av_client.get_daily_adjusted(symbol=sym, outputsize="compact")
                        if data is None or data.empty:
                            return None
                        df = data.reset_index().rename(columns={"date": "timestamp", "5. adjusted close": "adj_close", "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close", "6. volume": "volume"})
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        return df[["timestamp", "open", "high", "low", "close", "adj_close", "volume"]]
                    except Exception:
                        logger.exception("AlphaVantage fetch failed for %s", sym)
                        return None

                loop = asyncio.get_event_loop()
                df = await loop.run_in_executor(_get_threadpool(), _av_fetch)
                if df is not None and not df.empty:
                    _mem_cache.set(cache_key, df.to_json(), ttl=1800)
                    _file_cache.set(cache_key, json.loads(df.to_json()), ttl=getattr(cfg, "PRICE_CACHE_TTL", 1800))
                    try:
                        loop.run_in_executor(_get_threadpool(), lambda: db.save_price_history(cfg, sym, df))
                    except Exception:
                        logger.exception("Failed to persist alpha vantage price history to DB for %s", sym)
                    return df
        except Exception:
            logger.exception("AlphaVantage path failed for %s", sym)

    # 3) CoinGecko for crypto
    if is_crypto and _HAS_COINGECKO:
        try:
            cg = _SERVICES.get("cg") or CoinGeckoAPI()
            # coin gecko uses ids (e.g., 'bitcoin') not tickers. cfg can provide mapping CRYPTO_ID_MAP
            id_map = getattr(cfg, "CRYPTO_ID_MAP", {})
            cg_id = id_map.get(sym) or sym.lower()
            # use market_chart API for vs_currency 'usd' and days param
            days = "max" if period == "max" else str(int(max(30, 30 * (1 if period == "1mo" else 12))))
            data = cg.get_coin_market_chart_by_id(cg_id, vs_currency="usd", days=days)
            prices = data.get("prices", [])
            if not prices:
                return None
            rows = []
            for ts_ms, price in prices:
                ts = datetime.utcfromtimestamp(ts_ms / 1000.0)
                rows.append({"timestamp": ts, "open": None, "high": None, "low": None, "close": price, "adj_close": price, "volume": None})
            df = pd.DataFrame(rows)
            _mem_cache.set(cache_key, df.to_json(), ttl=3600)
            _file_cache.set(cache_key, json.loads(df.to_json()), ttl=getattr(cfg, "PRICE_CACHE_TTL", 3600))
            try:
                loop = asyncio.get_event_loop()
                loop.run_in_executor(_get_threadpool(), lambda: db.save_price_history(cfg, sym, df))
            except Exception:
                logger.exception("Failed to persist coingecko data to DB for %s", sym)
            return df
        except Exception:
            logger.exception("CoinGecko fetch failed for %s", sym)

    logger.debug("All price providers failed for %s", sym)
    return None


async def fetch_fundamentals(cfg: Any, symbol: str) -> Dict[str, Any]:
    """
    Fetch fundamental metrics for a symbol.

    Sources:
    - yfinance.Ticker.info (fast & free)
    - AlphaVantage fundamentals endpoints (if available)
    - Returns normalized dict with keys used elsewhere:
        revenue_yoy, gross_margin, free_cash_flow, debt_to_equity, roe, pe, ps, pb
    """
    if not symbol:
        return {}
    sym = symbol.upper()
    cache_key = f"fund:{sym}"
    cached = _mem_cache.get(cache_key)
    if cached is not None:
        return cached

    # 1) yfinance
    if _HAS_YFINANCE:
        try:
            def _yf_info():
                try:
                    tk = yf.Ticker(sym)
                    info = tk.info or {}
                    # extract key fields safely
                    revenue = info.get("revenueGrowth") or info.get("revenueGrowthTTM") or info.get("revenueGrowth", None)
                    gross_margin = info.get("grossMargins") or info.get("grossMarginsTTM")
                    fcf = info.get("freeCashflow") or info.get("freeCashFlow")
                    debt = info.get("debtToEquity") or info.get("debtToEquityRatio")
                    roe = info.get("returnOnEquity") or info.get("roe")
                    pe = info.get("trailingPE") or info.get("pe")
                    ps = info.get("priceToSalesTrailing12Months") or info.get("ps")
                    pb = info.get("priceToBook") or info.get("pb")
                    return {
                        "revenue_yoy": revenue if revenue is not None else 0.0,
                        "gross_margin": gross_margin if gross_margin is not None else 0.0,
                        "free_cash_flow": fcf if fcf is not None else 0.0,
                        "debt_to_equity": debt if debt is not None else 0.0,
                        "roe": roe if roe is not None else 0.0,
                        "pe": pe,
                        "ps": ps,
                        "pb": pb,
                        "raw": info,
                    }
                except Exception:
                    logger.exception("yfinance info fetch failed for %s", sym)
                    return {}

            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(_get_threadpool(), _yf_info)
            if res:
                _mem_cache.set(cache_key, res, ttl=getattr(cfg, "FUND_CACHE_TTL", 3600))
                return res
        except Exception:
            logger.exception("yfinance fundamentals path failed for %s", sym)

    # 2) AlphaVantage (OVERVIEW)
    av = _SERVICES.get("alpha_ts")
    if av is not None:
        try:
            # alpha vantage TimeSeries client doesn't provide company overview in same class; use requests to query API
            av_key = getattr(cfg, "ALPHAVANTAGE_API_KEY", None)
            if av_key and _allow_rate("alpha", getattr(cfg, "ALPHAVANTAGE_RATE_PER_MIN", 5)):
                url = "https://www.alphavantage.co/query"
                params = {"function": "OVERVIEW", "symbol": sym, "apikey": av_key}
                # try async http
                resp = await _http_get_async(url, params=params)
                if resp and resp.get("json"):
                    info = resp["json"]
                    # map fields conservatively
                    def _getf(k):
                        v = info.get(k)
                        try:
                            return float(v) if v is not None else None
                        except Exception:
                            return None
                    res = {
                        "revenue_yoy": None,
                        "gross_margin": None,
                        "free_cash_flow": _getf("FreeCashFlow"),
                        "debt_to_equity": _getf("DebtToEquity"),
                        "roe": _getf("ReturnOnEquityTTM"),
                        "pe": _getf("PERatio"),
                        "ps": None,
                        "pb": _getf("PriceToBookRatio"),
                        "raw": info,
                    }
                    _mem_cache.set(cache_key, res, ttl=getattr(cfg, "FUND_CACHE_TTL", 3600))
                    return res
        except Exception:
            logger.exception("AlphaVantage fundamentals failed for %s", sym)

    # fallback empty
    _mem_cache.set(cache_key, {}, ttl=60)
    return {}


# -----------------------
# RSS ingestion
# -----------------------
async def fetch_rss(cfg: Any, url: str, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """
    Fetch single RSS/Atom feed and parse items.

    Returns list of items: {title, link, published (iso), summary/text, id}
    """
    if not url:
        return []
    cache_key = f"rss:{url}"
    cached = _mem_cache.get(cache_key)
    if cached is not None and since is None:
        return cached
    # Try async fetch if available
    resp = await _http_get_async(url, timeout=15)
    feed_text = None
    if resp and "text" in resp and resp["text"]:
        feed_text = resp["text"]
    elif _HAS_REQUESTS:
        # fallback sync
        r = _http_get_sync(url, timeout=15)
        if r and r.get("text"):
            feed_text = r["text"]
    items: List[Dict[str, Any]] = []
    if feed_text and _HAS_FEEDPARSER:
        try:
            parsed = feedparser.parse(feed_text)
            for e in parsed.entries:
                pub = None
                if getattr(e, "published", None):
                    try:
                        pub = datetime(*e.published_parsed[:6]).isoformat()
                    except Exception:
                        pub = getattr(e, "published", None)
                item = {
                    "title": getattr(e, "title", "")[:500],
                    "link": getattr(e, "link", None),
                    "published": pub,
                    "id": getattr(e, "id", None) or getattr(e, "link", None),
                    "text": getattr(e, "summary", "") or getattr(e, "content", [{"value": ""}])[0].get("value", ""),
                    "raw": e,
                }
                # filter by 'since' if provided
                if since and item["published"]:
                    try:
                        pdt = datetime.fromisoformat(item["published"])
                        if pdt < since:
                            continue
                    except Exception:
                        pass
                items.append(item)
        except Exception:
            logger.exception("feedparser failed for %s", url)
    else:
        # fallback: try to extract simple <item> tags manually (very rough)
        logger.debug("No feedparser or empty content for %s", url)
    # cache
    _mem_cache.set(cache_key, items, ttl=getattr(cfg, "RSS_CACHE_TTL", 1800))
    return items


async def fetch_all_rss(cfg: Any) -> List[Dict[str, Any]]:
    """
    Fetch all configured RSS feeds in cfg.MONITOR_RSS_FEEDS (list).
    Returns flattened list of items.
    """
    feeds = getattr(cfg, "MONITOR_RSS_FEEDS", []) or []
    tasks = []
    for f in feeds:
        tasks.append(fetch_rss(cfg, f))
    results = []
    if tasks:
        res = await asyncio.gather(*tasks, return_exceptions=True)
        for r in res:
            if isinstance(r, Exception):
                logger.exception("fetch_rss task failed: %s", r)
            elif isinstance(r, list):
                results.extend(r)
    return results


# -----------------------
# Telegram ingestion helpers
# -----------------------
async def start_telethon_if_needed(cfg: Any) -> Optional[Any]:
    """
    Start Telethon client if configured and not started yet. Returns client or None.
    """
    client = _SERVICES.get("telethon_client")
    if client is None:
        return None
    try:
        if not client.is_connected():
            await client.start()
        return client
    except Exception:
        logger.exception("Failed to start Telethon client")
        return None


async def fetch_all_telegram(cfg: Any) -> List[Dict[str, Any]]:
    """
    Fetch recent messages from configured telegram channels (MONITOR_TELEGRAM_CHANNELS).
    Requires Telethon client credentials in cfg (TELEGRAM_API_ID/HASH and session).
    Returns list of {text, timestamp, channel}
    """
    channels = getattr(cfg, "MONITOR_TELEGRAM_CHANNELS", []) or []
    if not channels:
        return []
    client = _SERVICES.get("telethon_client")
    if client is None:
        logger.debug("Telethon client not configured")
        return []
    # ensure started
    try:
        await start_telethon_if_needed(cfg)
    except Exception:
        logger.exception("Telethon start failed")
        return []
    results = []
    try:
        # Telethon is synchronous in interface here; run in threadpool to avoid blocking
        def _fetch():
            out = []
            try:
                for ch in channels:
                    try:
                        # accept both username or channel id
                        msgs = client.get_messages(ch, limit=50)
                        for m in msgs:
                            out.append({"text": m.message or "", "timestamp": getattr(m, "date", None).isoformat() if getattr(m, "date", None) else None, "channel": ch})
                    except Exception:
                        logger.exception("Failed fetch messages for channel %s", ch)
                return out
            except Exception:
                logger.exception("Telethon fetch thread failed")
                return out

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(_get_threadpool(), _fetch)
    except Exception:
        logger.exception("fetch_all_telegram failed")
    return results


async def fetch_recent_messages_for_symbol(cfg: Any, symbol: str, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """
    Fetch recent Telegram messages and filter those mentioning the symbol.
    Provided for watch loops to quickly check messages relevant to a ticker.
    """
    items = await fetch_all_telegram(cfg)
    out = []
    sym = symbol.upper()
    for it in items:
        txt = it.get("text", "")
        if sym in txt.upper():
            ts = it.get("timestamp")
            if since and ts:
                try:
                    tparsed = datetime.fromisoformat(ts)
                    if tparsed < since:
                        continue
                except Exception:
                    pass
            out.append(it)
    return out


# -----------------------
# SEC EDGAR helpers
# -----------------------
async def fetch_sec_filings(cfg: Any, symbol: str, count: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch recent SEC filings via EDGAR RSS or /Archives API.

    Returns list of {accession, filing_type, date, summary, link}
    """
    if not symbol:
        return []
    # Use SEC RSS feed for company filings if available via search query
    # fallback: use SEC EDGAR company filings search (slow)
    try:
        base_rss = getattr(cfg, "SEC_RSS_URL", "https://www.sec.gov/Archives/edgar/usgaap.rss")
        # fetch and then filter by symbol in text/title
        items = await fetch_rss(cfg, base_rss)
        out = []
        for it in items:
            title = it.get("title", "")
            if symbol.upper() in title.upper() or symbol.upper() in it.get("text", "").upper():
                out.append(it)
            if len(out) >= count:
                break
        return out
    except Exception:
        logger.exception("fetch_sec_filings failed for %s", symbol)
        return []


# -----------------------
# Misc helpers used by core
# -----------------------
async def health_check(cfg: Any) -> Dict[str, Any]:
    """
    Quick health check of configured external providers (AlphaVantage, CoinGecko, Telegram).
    Returns statuses.
    """
    res = {"time": datetime.utcnow().isoformat(), "providers": {}}
    res["providers"]["yfinance"] = {"available": _HAS_YFINANCE}
    res["providers"]["alpha_vantage"] = {"available": _SERVICES.get("alpha_ts") is not None}
    res["providers"]["coingecko"] = {"available": _SERVICES.get("cg") is not None}
    res["providers"]["telethon"] = {"available": _SERVICES.get("telethon_client") is not None}
    return res