# app/services.py
"""
Services layer: API clients, ingestion and helpers.

Provides:
 - fetch_price_history(cfg, symbol) -> pandas.DataFrame | None
 - fetch_fundamentals(cfg, symbol) -> dict | None
 - fetch_all_rss(cfg) -> list[dict]
 - fetch_rss(cfg, url, since=None) -> list[dict]
 - fetch_all_telegram(cfg) -> list[dict]
 - fetch_recent_messages_for_symbol(cfg, symbol, since=None) -> list[dict]

Design goals:
 - Works both sync and async: prefer aiohttp if available, else run blocking requests in threadpool.
 - Simple caching (in-memory + file) with TTL.
 - Respect basic rate limits.
 - Fail gracefully and log errors.
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
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("app.services")
logger.addHandler(logging.NullHandler())

# Optional libs (best-effort imports)
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

# Telethon optional for Telegram ingestion
try:
    from telethon import TelegramClient  # type: ignore
    _HAS_TELETHON = True
except Exception:
    TelegramClient = None  # type: ignore
    _HAS_TELETHON = False

# Local DB for optional persistence
import app.db as db  # type: ignore

# Thread pool (singleton)
_THREAD_POOL: Optional[ThreadPoolExecutor] = None
_THREAD_POOL_LOCK = threading.Lock()


def _get_threadpool(max_workers: int = 8) -> ThreadPoolExecutor:
    global _THREAD_POOL
    if _THREAD_POOL is None:
        with _THREAD_POOL_LOCK:
            if _THREAD_POOL is None:
                _THREAD_POOL = ThreadPoolExecutor(max_workers=max_workers)
    return _THREAD_POOL


# Simple persistent file cache (JSON)
class FileCache:
    def __init__(self, cache_dir: str = "./data/cache", default_ttl_sec: int = 6 * 3600):
        self.cache_dir = cache_dir
        self.ttl = default_ttl_sec
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = key.replace("/", "_").replace(":", "_").replace(" ", "_")
        return os.path.join(self.cache_dir, f"{safe}.json")

    def get(self, key: str) -> Optional[Any]:
        try:
            p = self._path(key)
            if not os.path.exists(p):
                return None
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            ts = datetime.fromisoformat(obj.get("_cached_at"))
            ttl = int(obj.get("_ttl", self.ttl))
            if (datetime.utcnow() - ts).total_seconds() > ttl:
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


# Simple in-memory TTL cache
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

# naive per-provider rate-limiter (token-bucket like)
_RATE_STATE: Dict[str, Dict[str, Any]] = {}
_RATE_LOCK = threading.Lock()


def _allow_rate(provider: str, calls_per_min: int = 5) -> bool:
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


# Global clients registry
_SERVICES: Dict[str, Any] = {
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
    Initialize optional clients (AlphaVantage, CoinGecko, Telethon, aiohttp session).
    This function is synchronous to keep compatibility; core may call it via maybe_await.
    """
    logger.info("services.initialize starting")
    # AlphaVantage
    try:
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
    except Exception:
        logger.exception("AlphaVantage init error")
        _SERVICES["alpha_ts"] = None

    # CoinGecko
    try:
        if _HAS_COINGECKO:
            _SERVICES["cg"] = CoinGeckoAPI()
            logger.info("CoinGecko client ready")
        else:
            _SERVICES["cg"] = None
    except Exception:
        logger.exception("CoinGecko init error")
        _SERVICES["cg"] = None

    # Telethon (prepare only)
    try:
        tg_api_id = getattr(cfg, "TELEGRAM_API_ID", None)
        tg_api_hash = getattr(cfg, "TELEGRAM_API_HASH", None)
        if tg_api_id and tg_api_hash and _HAS_TELETHON:
            session_name = getattr(cfg, "TELETHON_SESSION", "services_telethon_session")
            try:
                client = TelegramClient(session_name, tg_api_id, tg_api_hash)
                _SERVICES["telethon_client"] = client
                logger.info("Telethon client prepared (not connected)")
            except Exception:
                logger.exception("Failed to prepare Telethon client")
                _SERVICES["telethon_client"] = None
        else:
            _SERVICES["telethon_client"] = None
    except Exception:
        logger.exception("Telethon init error")
        _SERVICES["telethon_client"] = None

    # aiohttp session (create in running event loop if available)
    # aiohttp session (создаём лениво, без run_until_complete)
try:
    if _HAS_AIOHTTP:
        # всегда инициализируем как None, создаём сессию при первом запросе
        _SERVICES["aiohttp_session"] = None
        logger.info("aiohttp session will be created lazily on first use")
    else:
        _SERVICES["aiohttp_session"] = None
except Exception:
    logger.exception("aiohttp init error")
    _SERVICES["aiohttp_session"] = None


def shutdown(cfg: Any = None) -> None:
    """Shutdown clients and thread pool."""
    logger.info("services.shutdown starting")
    # aiohttp
    try:
        sess = _SERVICES.get("aiohttp_session")
        if sess is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop and loop.is_running():
                    loop.run_until_complete(sess.close())
                else:
                    # create temporary loop
                    asyncio.run(sess.close())
            except Exception:
                logger.exception("Failed to close aiohttp session")
            _SERVICES["aiohttp_session"] = None
    except Exception:
        logger.exception("Error shutting down aiohttp")

    # Telethon disconnect
    try:
        client = _SERVICES.get("telethon_client")
        if client:
            try:
                loop = asyncio.get_event_loop()
                if loop and loop.is_running():
                    asyncio.create_task(client.disconnect())
                else:
                    asyncio.run(client.disconnect())
            except Exception:
                logger.exception("Failed to disconnect Telethon client")
            _SERVICES["telethon_client"] = None
    except Exception:
        logger.exception("Error shutting down telethon")

    # threadpool
    global _THREAD_POOL
    try:
        if _THREAD_POOL is not None:
            _THREAD_POOL.shutdown(wait=True)
            _THREAD_POOL = None
    except Exception:
        logger.exception("Failed to shutdown threadpool")

    logger.info("services.shutdown done")


# -----------------------
# HTTP helpers
# -----------------------
async def _http_get_async(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """GET via aiohttp (if available), returns dict with status, text, json (json may be None)."""
    if not _HAS_AIOHTTP:
        return await asyncio.get_event_loop().run_in_executor(_get_threadpool(), lambda: _http_get_sync(url, params, headers, timeout))

    sess = _SERVICES.get("aiohttp_session")
    if sess is None:
        # use a temporary session (context-managed)
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url, params=params or {}, headers=headers or {}, timeout=timeout) as resp:
                    text = await resp.text()
                    js = None
                    try:
                        js = await resp.json()
                    except Exception:
                        js = None
                    return {"status": resp.status, "text": text, "json": js}
        except Exception:
            logger.exception("aiohttp temporary session GET failed for %s", url)
            return None

    try:
        async with sess.get(url, params=params or {}, headers=headers or {}, timeout=timeout) as resp:
            text = await resp.text()
            js = None
            try:
                js = await resp.json()
            except Exception:
                js = None
            return {"status": resp.status, "text": text, "json": js}
    except Exception:
        logger.exception("aiohttp GET failed for %s", url)
        return None


def _http_get_sync(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """Sync GET using requests, returns dict with status, text, json (json may be None)."""
    if not _HAS_REQUESTS:
        logger.error("requests library not installed; cannot perform HTTP GET for %s", url)
        return None
    try:
        resp = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
        text = resp.text
        js = None
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
    Fetch price history for a symbol. Returns pandas DataFrame with columns:
    ['timestamp','open','high','low','close','adj_close','volume'] or None.
    Caching: in-memory + file-cache (we store JSON string from df.to_json()).
    """
    if not symbol:
        return None
    sym = symbol.upper()
    cache_key = f"price:{sym}:{period}:{interval}"
    # 1) mem cache (we store JSON string)
    cached = _mem_cache.get(cache_key)
    if cached is not None:
        try:
            return pd.read_json(cached)
        except Exception:
            logger.exception("Failed to read DataFrame from mem cache for %s", sym)
    # 2) file cache
    fc = _file_cache.get(cache_key)
    if fc is not None:
        try:
            # fc is a JSON string produced by df.to_json()
            df = pd.read_json(fc)
            # warm mem cache
            _mem_cache.set(cache_key, df.to_json(), ttl=3600)
            return df
        except Exception:
            logger.exception("Failed to read DataFrame from file cache for %s", sym)

    # detect crypto (simple heuristics + cfg overrides)
    is_crypto = False
    crypto_list = getattr(cfg, "KNOWN_CRYPTO_SYMBOLS", ["BTC", "ETH", "USDT", "XRP", "LTC"])
    if sym in crypto_list or getattr(cfg, "FORCE_CRYPTO", False):
        is_crypto = True

    # 1) yfinance (blocking) -> do in threadpool
    if not is_crypto and _HAS_YFINANCE:
        loop = asyncio.get_event_loop()
        def _yf_fetch():
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(period=period, interval=interval, actions=False)
                if hist is None or hist.empty:
                    return None
                hist = hist.reset_index()
                hist = hist.rename(columns={"Date": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", "Adj Close": "adj_close"})
                if "adj_close" not in hist.columns and "Close" in hist.columns:
                    hist["adj_close"] = hist["close"]
                hist["timestamp"] = pd.to_datetime(hist["timestamp"])
                return hist[["timestamp", "open", "high", "low", "close", "adj_close", "volume"]]
            except Exception:
                logger.exception("yfinance fetch failed for %s", sym)
                return None
        df = await loop.run_in_executor(_get_threadpool(), _yf_fetch)
        if df is not None and not df.empty:
            try:
                json_str = df.to_json()
                _mem_cache.set(cache_key, json_str, ttl=3600)
                _file_cache.set(cache_key, json_str, ttl=getattr(cfg, "PRICE_CACHE_TTL", 3600))
            except Exception:
                logger.exception("Failed to cache price history for %s", sym)
            # persist to DB in background
            try:
                loop.run_in_executor(_get_threadpool(), lambda: db.save_price_history(cfg, sym, df))
            except Exception:
                logger.exception("Failed to persist price history for %s", sym)
            return df

    # 2) AlphaVantage (pandas output) fallback
    av_client = _SERVICES.get("alpha_ts")
    if av_client is not None and _HAS_ALPHA:
        try:
            if not _allow_rate("alpha", getattr(cfg, "ALPHAVANTAGE_RATE_PER_MIN", 5)):
                logger.debug("AlphaVantage rate limit reached for %s", sym)
            else:
                loop = asyncio.get_event_loop()
                def _av_fetch():
                    try:
                        # client.get_daily_adjusted -> (DataFrame, meta) when using pandas format
                        data, meta = av_client.get_daily_adjusted(symbol=sym, outputsize="full")
                        if data is None or data.empty:
                            return None
                        df = data.reset_index().rename(columns={"date": "timestamp", "5. adjusted close": "adj_close"})
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        # normalize columns if needed
                        # rename or map columns to expected ones:
                        # alpha columns can differ; try mapping common names
                        col_map = {}
                        for c in df.columns:
                            lc = c.lower()
                            if "open" in lc and "adjusted" not in lc:
                                col_map[c] = "open"
                            if "high" in lc:
                                col_map[c] = "high"
                            if "low" in lc:
                                col_map[c] = "low"
                            if "close" in lc and "adjusted" not in lc:
                                col_map[c] = "close"
                            if "adjusted" in lc:
                                col_map[c] = "adj_close"
                            if "volume" in lc:
                                col_map[c] = "volume"
                        df = df.rename(columns=col_map)
                        # ensure all expected columns present
                        for col in ["open","high","low","close","adj_close","volume"]:
                            if col not in df.columns:
                                df[col] = None
                        return df[["timestamp", "open","high","low","close","adj_close","volume"]]
                    except Exception:
                        logger.exception("AlphaVantage fetch failed for %s", sym)
                        return None
                df = await loop.run_in_executor(_get_threadpool(), _av_fetch)
                if df is not None and not df.empty:
                    try:
                        json_str = df.to_json()
                        _mem_cache.set(cache_key, json_str, ttl=3600)
                        _file_cache.set(cache_key, json_str, ttl=getattr(cfg, "PRICE_CACHE_TTL", 3600))
                    except Exception:
                        logger.exception("Failed to cache price history (alpha) for %s", sym)
                    try:
                        loop.run_in_executor(_get_threadpool(), lambda: db.save_price_history(cfg, sym, df))
                    except Exception:
                        logger.exception("Failed to persist price history (alpha) for %s", sym)
                    return df
        except Exception:
            logger.exception("AlphaVantage path error for %s", sym)

    # 3) CoinGecko for cryptos (simple mapping)
    if is_crypto and _HAS_COINGECKO:
        try:
            cg = _SERVICES.get("cg")
            if cg is None:
                cg = CoinGeckoAPI()
                _SERVICES["cg"] = cg
            # CoinGecko uses ids (e.g. 'bitcoin'), mapping from symbol may be required.
            # Keep simple: try market_chart_by_id if user provided CRYPTO_IDS mapping
            cg_map = getattr(cfg, "CRYPTO_ID_MAP", {})
            coin_id = cg_map.get(sym)
            if coin_id:
                if not _allow_rate("coingecko", getattr(cfg, "COINGECKO_RATE_PER_MIN", 30)):
                    logger.debug("CoinGecko rate limit reached for %s", sym)
                else:
                    res = cg.get_coin_market_chart_by_id(coin_id, vs_currency="usd", days=365)
                    if res and "prices" in res:
                        # convert list of [ts, price] to DataFrame daily approximations
                        prices = res["prices"]
                        df = pd.DataFrame(prices, columns=["ts", "price"])
                        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
                        df = df.set_index("timestamp").resample("1D").first().reset_index()
                        df["open"] = df["price"]
                        df["high"] = df["price"]
                        df["low"] = df["price"]
                        df["close"] = df["price"]
                        df["adj_close"] = df["price"]
                        df["volume"] = None
                        json_str = df.to_json()
                        _mem_cache.set(cache_key, json_str, ttl=3600)
                        _file_cache.set(cache_key, json_str, ttl=getattr(cfg, "PRICE_CACHE_TTL", 3600))
                        try:
                            loop = asyncio.get_event_loop()
                            loop.run_in_executor(_get_threadpool(), lambda: db.save_price_history(cfg, sym, df))
                        except Exception:
                            logger.exception("Failed to persist crypto price history for %s", sym)
                        return df[["timestamp","open","high","low","close","adj_close","volume"]]
        except Exception:
            logger.exception("CoinGecko path failed for %s", sym)

    logger.debug("No price data found for %s", sym)
    return None


# -----------------------
# Placeholder for other service functions
# -----------------------
# fetch_fundamentals, fetch_all_rss, fetch_rss, fetch_all_telegram, fetch_recent_messages_for_symbol
# These functions depend on external providers and project choices; provide safe stubs that
# return None or empty list when not implemented so core can function.

def fetch_fundamentals(cfg: Any, symbol: str) -> Optional[Dict[str, Any]]:
    """Return fundamentals dict or None. Implement provider-specific logic here."""
    # Example: try yfinance fast info if available
    try:
        if _HAS_YFINANCE:
            try:
                t = yf.Ticker(symbol)
                info = t.info if hasattr(t, "info") else {}
                return info or None
            except Exception:
                logger.debug("yfinance fundamentals failed for %s", symbol)
                return None
    except Exception:
        logger.exception("fetch_fundamentals error for %s", symbol)
    return None


def fetch_all_rss(cfg: Any) -> List[Dict[str, Any]]:
    """Fetch all configured RSS feeds; returns list of item dicts."""
    feeds = getattr(cfg, "MONITOR_RSS_FEEDS", []) or []
    out: List[Dict[str, Any]] = []
    if not feeds or not _HAS_FEEDPARSER:
        return out
    try:
        for f in feeds:
            try:
                parsed = feedparser.parse(f)
                for e in parsed.entries:
                    out.append({"title": getattr(e, "title", ""), "summary": getattr(e, "summary", ""), "published": getattr(e, "published", None), "link": getattr(e, "link", None)})
            except Exception:
                logger.exception("Failed to fetch RSS feed %s", f)
    except Exception:
        logger.exception("fetch_all_rss top-level error")
    return out


def fetch_rss(cfg: Any, url: str, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Fetch one RSS feed and optionally filter by 'since' datetime."""
    items = fetch_all_rss(cfg)
    if since:
        return [it for it in items if it.get("published") and datetime.fromisoformat(it.get("published")) >= since]
    return items


def fetch_all_telegram(cfg: Any) -> List[Dict[str, Any]]:
    """Fetch recent telegram messages using Telethon client (if available)."""
    client = _SERVICES.get("telethon_client")
    if client is None or not _HAS_TELETHON:
        return []
    # Telethon usage requires event loop and proper connection; keep simple stub
    try:
        # Caller (Orchestrator) may start/connect client; here we attempt a quick fetch if connected
        if getattr(client, "is_connected", False):
            # Example placeholder (real implementation requires telethon API usage):
            return []
    except Exception:
        logger.exception("fetch_all_telegram error")
    return []


def fetch_recent_messages_for_symbol(cfg: Any, symbol: str, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Try to fetch recent messages from telegram/rss and return those mentioning symbol."""
    out: List[Dict[str, Any]] = []
    try:
        # first RSS
        if getattr(cfg, "ENABLE_RSS_INGEST", False):
            try:
                rss_items = fetch_all_rss(cfg)
                for it in rss_items:
                    text = (it.get("title") or "") + " " + (it.get("summary") or "")
                    if symbol.upper() in text.upper():
                        out.append({"text": text, "timestamp": it.get("published"), "extra": it})
            except Exception:
                logger.exception("fetch_recent_messages_for_symbol RSS error for %s", symbol)
        # telegram messages (if telethon is available)
        if getattr(cfg, "ENABLE_TELEGRAM_INGEST", False) and _HAS_TELETHON:
            try:
                tg = fetch_all_telegram(cfg)
                for it in tg:
                    text = it.get("text", "")
                    if symbol.upper() in text.upper():
                        out.append({"text": text, "timestamp": it.get("timestamp"), "extra": it})
            except Exception:
                logger.exception("fetch_recent_messages_for_symbol TG error for %s", symbol)
    except Exception:
        logger.exception("fetch_recent_messages_for_symbol top error for %s", symbol)
    # optionally filter by since
    if since:
        try:
            filtered = []
            for it in out:
                ts = it.get("timestamp")
                if ts is None:
                    continue
                try:
                    tdt = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
                    if tdt >= since:
                        filtered.append(it)
                except Exception:
                    continue
            return filtered
        except Exception:
            logger.exception("Filtering by since failed")
    return out