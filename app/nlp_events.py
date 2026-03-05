# app/nlp_events.py
"""
NLP and Event processing pipeline for quasihedge-bot.

Enhancements in this version:
- Parallel embedding computation for batches (ThreadPoolExecutor)
- Background worker (thread + queue) for event impact calculation (non-blocking ingestion)
- Improved datetime parsing (dateutil) and timezone normalization to UTC
- Better relevance scoring via spaCy NER (if available) with regex fallback
- TF-IDF fallback embedding (sklearn) instead of naive hashing
- Optional lightweight ML event-type classifier hooks (trainable offline)
- Optional Prometheus metrics (if prometheus_client installed)
- Robust logging, type hints, docstrings, and defensive checks
"""

from __future__ import annotations

import json
import logging
import math
import queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Optional libraries
try:
    from dateutil import parser as dateutil_parser  # type: ignore
    _HAS_DATEUTIL = True
except Exception:
    dateutil_parser = None  # type: ignore
    _HAS_DATEUTIL = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
    _HAS_VADER = True
except Exception:
    SentimentIntensityAnalyzer = None  # type: ignore
    _HAS_VADER = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    TfidfVectorizer = None  # type: ignore
    LogisticRegression = None  # type: ignore
    _HAS_SKLEARN = False

try:
    import spacy  # type: ignore
    _HAS_SPACY = True
except Exception:
    spacy = None  # type: ignore
    _HAS_SPACY = False

# Optional Prometheus metrics
try:
    from prometheus_client import Counter  # type: ignore

    METRICS_ENABLED = True
except Exception:
    Counter = None  # type: ignore
    METRICS_ENABLED = False

# Local DB module (assumed provided)
import app.db as db

logger = logging.getLogger("app.nlp_events")
logger.addHandler(logging.NullHandler())

# -----------------------
# Constants / heuristics
# -----------------------
_SYMBOL_REGEX = re.compile(r"(?:\$)?\b([A-Z]{1,5})\b")
_PAREN_SYMBOL_REGEX = re.compile(r"\(([A-Z0-9]{1,6})\)")
_TICKER_CASHTAG_REGEX = re.compile(r"\$([A-Z]{1,6})\b")
_DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

_EVENT_KEYWORDS = {
    "earnings": ["earnings", "q1", "q2", "q3", "q4", "quarterly", "eps", "revenue", "guidance"],
    "contract": ["contract", "deal", "agreement", "signed with", "partner", "partnership"],
    "product": ["launch", "release", "ship", "product", "version", "roadmap"],
    "merger": ["merger", "acquisition", "acquire", "buyout"],
    "regulation": ["regulation", "ban", "sanction", "fine", "law", "legislation"],
    "geopolitical": ["war", "invasion", "sanction", "embargo", "attack", "conflict"],
    "insider": ["insider", "13f", "insider buy", "insider sell", "bought shares", "sold shares"],
    "rumor": ["rumor", "leak", "unconfirmed", "reported"],
    "listing": ["ipo", "listing", "listed", "delist", "exchange listing"],
}

_POSITIVE_WORDS = {"upgrade", "beat", "gain", "growth", "profit", "surge", "record", "positive"}
_NEGATIVE_WORDS = {"miss", "missed", "drop", "loss", "decline", "bankrupt", "lawsuit", "scandal", "delay"}

# -----------------------
# Prometheus metrics (optional)
# -----------------------
if METRICS_ENABLED:
    MET_EMBEDDINGS_PROCESSED = Counter("nlp_embeddings_processed_total", "Total embeddings processed")
    MET_EVENTS_PROCESSED = Counter("nlp_events_processed_total", "Total events processed")
    MET_EMBED_FAIL = Counter("nlp_embedding_failures_total", "Embedding failures")
    MET_IMPACT_ENQUEUED = Counter("nlp_impact_tasks_enqueued_total", "Event impact tasks queued")
else:
    # no-op placeholders
    class _Noop:
        def inc(self, *_, **__):
            pass

    MET_EMBEDDINGS_PROCESSED = MET_EVENTS_PROCESSED = MET_EMBED_FAIL = MET_IMPACT_ENQUEUED = _Noop()


# -----------------------
# Models container
# -----------------------
@dataclass
class Models:
    """Container for optional models loaded lazily."""

    embed_model: Optional[Any] = None
    tfidf: Optional[Any] = None
    tfidf_ready: bool = False
    vader: Optional[Any] = None
    event_clf: Optional[Any] = None  # sklearn classifier

    # concurrency helpers
    _embed_executor: Optional[ThreadPoolExecutor] = None

    @classmethod
    def ensure_embed_executor(cls, max_workers: int = 4):
        if cls._embed_executor is None:
            cls._embed_executor = ThreadPoolExecutor(max_workers=max_workers)
        return cls._embed_executor

    @classmethod
    def load_embedding_model(cls, model_name: str = _DEFAULT_EMBED_MODEL):
        """Load sentence-transformers model if available."""
        if cls.embed_model is not None:
            return cls.embed_model
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                cls.embed_model = SentenceTransformer(model_name)
                logger.info("Loaded embedding model: %s", model_name)
            except Exception:
                cls.embed_model = None
                logger.exception("Failed to load sentence-transformers model '%s'", model_name)
        else:
            logger.info("sentence-transformers not installed; using TF-IDF fallback if available")
            cls.embed_model = None
        return cls.embed_model

    @classmethod
    def load_vader(cls):
        """Load VADER if available."""
        if cls.vader is not None:
            return cls.vader
        if _HAS_VADER:
            try:
                cls.vader = SentimentIntensityAnalyzer()
                logger.info("Loaded VADER sentiment analyzer")
            except Exception:
                cls.vader = None
                logger.exception("Failed to initialize VADER")
        else:
            cls.vader = None
        return cls.vader

    @classmethod
    def ensure_tfidf(cls, texts: Sequence[str]):
        """Fit TF-IDF on provided corpus if sklearn is available."""
        if cls.tfidf is not None and cls.tfidf_ready:
            return cls.tfidf
        if not _HAS_SKLEARN:
            cls.tfidf = None
            cls.tfidf_ready = False
            logger.debug("sklearn not available; TF-IDF not usable")
            return None
        try:
            cls.tfidf = TfidfVectorizer(max_features=1024, ngram_range=(1, 2), stop_words="english")
            cls.tfidf.fit(list(texts))
            cls.tfidf_ready = True
            logger.info("TF-IDF fitted with %d features", len(cls.tfidf.get_feature_names_out()))
            return cls.tfidf
        except Exception:
            cls.tfidf = None
            cls.tfidf_ready = False
            logger.exception("TF-IDF fit failed")
            return None

    @classmethod
    def load_event_classifier(cls, path: Optional[str] = None):
        """
        Load a pre-trained sklearn classifier (LogisticRegression) if available.
        Path points to joblib/pickle file. If not present, event_clf remains None.
        """
        if cls.event_clf is not None:
            return cls.event_clf
        if not _HAS_SKLEARN:
            logger.debug("sklearn not available; event classifier disabled")
            cls.event_clf = None
            return None
        if path:
            try:
                import joblib  # type: ignore

                cls.event_clf = joblib.load(path)
                logger.info("Loaded event classifier from %s", path)
            except Exception:
                cls.event_clf = None
                logger.exception("Failed to load event classifier from %s", path)
        return cls.event_clf


# -----------------------
# Background worker for event-impact calculation
# -----------------------
_IMPACT_QUEUE: "queue.Queue[int]" = queue.Queue()
_IMPACT_WORKER_THREAD: Optional[threading.Thread] = None
_IMPACT_WORKER_RUNNING = threading.Event()


def _impact_worker(cfg: Any):
    """
    Worker thread that consumes event IDs and runs calculate_event_impact in background.
    """
    logger.info("Event-impact worker started")
    while _IMPACT_WORKER_RUNNING.is_set():
        try:
            event_id = _IMPACT_QUEUE.get(timeout=1)
        except queue.Empty:
            continue
        try:
            logger.debug("Background impact calc for event id=%s", event_id)
            db.calculate_event_impact(cfg, event_id)
        except Exception:
            logger.exception("Background calculate_event_impact failed for event id=%s", event_id)
        finally:
            _IMPACT_QUEUE.task_done()
    logger.info("Event-impact worker stopped")


def _start_impact_worker(cfg: Any):
    global _IMPACT_WORKER_THREAD
    if _IMPACT_WORKER_THREAD and _IMPACT_WORKER_THREAD.is_alive():
        return
    _IMPACT_WORKER_RUNNING.set()
    _IMPACT_WORKER_THREAD = threading.Thread(target=_impact_worker, args=(cfg,), daemon=True)
    _IMPACT_WORKER_THREAD.start()


def _stop_impact_worker():
    _IMPACT_WORKER_RUNNING.clear()
    if _IMPACT_WORKER_THREAD:
        _IMPACT_WORKER_THREAD.join(timeout=5)


# -----------------------
# Utility functions
# -----------------------
def _parse_datetime(dt_input: Any) -> datetime:
    """
    Robust datetime parser: supports datetime, ISO strings, and falls back to utcnow.
    Returns timezone-aware UTC datetime.
    """
    if isinstance(dt_input, datetime):
        if dt_input.tzinfo is None:
            return dt_input.replace(tzinfo=timezone.utc)
        return dt_input.astimezone(timezone.utc)
    if isinstance(dt_input, str):
        try:
            if _HAS_DATEUTIL:
                parsed = dateutil_parser.parse(dt_input)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            else:
                parsed = datetime.fromisoformat(dt_input)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
        except Exception:
            logger.exception("Failed to parse datetime string: %s", dt_input)
            return datetime.utcnow().replace(tzinfo=timezone.utc)
    # fallback
    return datetime.utcnow().replace(tzinfo=timezone.utc)


def _clean_text(text: str) -> str:
    """Basic cleaning of raw text."""
    if not text:
        return ""
    t = text.replace("\r", " ").replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _safe_normalize_vec(vec: Sequence[float]) -> Optional[np.ndarray]:
    """Return L2-normalized numpy array or None."""
    try:
        arr = np.asarray(vec, dtype=np.float64)
        if arr.size == 0:
            return None
        n = np.linalg.norm(arr)
        if n == 0 or math.isnan(n):
            return None
        return arr / n
    except Exception:
        return None


# -----------------------
# Text processing building blocks
# -----------------------
def extract_symbols(text: str) -> List[str]:
    """
    Extract candidate tickers from text. Returns list of uppercase tokens (no duplicates).
    """
    if not text:
        return []
    candidates = set()
    candidates.update([m.upper() for m in _TICKER_CASHTAG_REGEX.findall(text)])
    candidates.update([m.upper() for m in _PAREN_SYMBOL_REGEX.findall(text)])
    for m in _SYMBOL_REGEX.findall(text):
        if 1 <= len(m) <= 5:
            candidates.add(m.upper())
    # filter common false positives
    stopwords = {"CEO", "USD", "EUR", "SEC", "US", "UK"}
    symbols = [s for s in candidates if s not in stopwords]
    return sorted(symbols)


def classify_event_type_rule(text: str) -> str:
    """
    Rule-based event type classification.
    """
    if not text:
        return "other"
    lowered = text.lower()
    scores = {}
    for t, kws in _EVENT_KEYWORDS.items():
        cnt = 0
        for kw in kws:
            if kw in lowered:
                cnt += 1
        if cnt:
            scores[t] = cnt
    if not scores:
        return "other"
    # choose highest count, tie-breaker by len of keywords
    best = max(scores.items(), key=lambda x: (x[1], len(_EVENT_KEYWORDS.get(x[0], []))))
    return best[0]


def sentiment_score(text: str) -> float:
    """
    Compute sentiment in [-1,1].
    Uses VADER if present; otherwise simple lexicon approach.
    """
    if not text:
        return 0.0
    Models.load_vader()
    if Models.vader is not None:
        try:
            vs = Models.vader.polarity_scores(text)
            return float(vs.get("compound", 0.0))
        except Exception:
            logger.exception("VADER failed; using fallback sentiment")
    # fallback lexicon
    t = text.lower()
    pos = sum(1 for w in _POSITIVE_WORDS if w in t)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in t)
    if pos + neg == 0:
        return 0.0
    return float((pos - neg) / max(1, pos + neg))


def compute_relevance(text: str, symbol: Optional[str]) -> float:
    """
    Compute relevance [0,1] using spaCy NER if available, else regex heuristics.
    """
    if not text or not symbol:
        return 0.0
    sym = symbol.upper()
    # direct cashtag or parens
    if re.search(rf"\${re.escape(sym)}\b", text):
        return 1.0
    if re.search(rf"\(\s*{re.escape(sym)}\s*\)", text):
        return 1.0
    # spaCy NER-based: check if an ORG entity matches symbol or contains company name
    if _HAS_SPACY:
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("ORG", "PRODUCT", "GPE"):
                    if sym in ent.text.upper() or ent.text.upper() in sym:
                        return 0.95
            # if symbol uppercase occurs as separate token
            if re.search(rf"\b{re.escape(sym)}\b", text):
                return 0.85
        except Exception:
            logger.debug("spaCy entity recognition failed, falling back to regex")
    # regex fallback
    if re.search(rf"\b{re.escape(sym)}\b", text):
        return 0.85
    if sym.lower() in text.lower():
        return 0.5
    return 0.0


# -----------------------
# Embeddings: parallel processing and fallbacks
# -----------------------
def _embed_single(model, text: str) -> Optional[List[float]]:
    """Embed single text via sentence-transformers, TF-IDF fallback, or hash fallback."""
    if not text:
        return None
    # sentence-transformers preferred
    if model is not None:
        try:
            vec = model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
            a = np.asarray(vec, dtype=np.float64)
            n = np.linalg.norm(a)
            if n == 0 or math.isnan(n):
                return None
            return (a / n).astype(float).tolist()
        except Exception:
            logger.exception("SentenceTransformer embed failed for text; falling back")
    # TF-IDF fallback
    if Models.tfidf is not None and Models.tfidf_ready:
        try:
            arr = Models.tfidf.transform([text]).toarray()[0]
            a = np.asarray(arr, dtype=np.float64)
            n = np.linalg.norm(a)
            if n == 0 or math.isnan(n):
                return None
            return (a / n).astype(float).tolist()
        except Exception:
            logger.exception("TF-IDF transform failed; falling back to hashing")
    # Hash fallback (deterministic small vector)
    try:
        h = abs(hash(text)) % (10 ** 8)
        vec = np.array([((h >> (i * 8)) & 255) / 255.0 for i in range(16)], dtype=float)
        n = np.linalg.norm(vec)
        if n == 0 or math.isnan(n):
            return None
        return (vec / n).tolist()
    except Exception:
        return None


def embed_texts_parallel(texts: Sequence[str], cfg: Optional[Any] = None) -> List[Optional[List[float]]]:
    """
    Embed texts using a thread pool to parallelize embedding (useful for large batches).

    Returns list of embeddings (or None for failed).
    """
    if not texts:
        return []

    model_name = getattr(cfg, "EMBED_MODEL_NAME", _DEFAULT_EMBED_MODEL) if cfg is not None else _DEFAULT_EMBED_MODEL
    Models.load_embedding_model(model_name)
    model = Models.embed_model

    # ensure TF-IDF ready if embed model absent
    if model is None and _HAS_SKLEARN and not Models.tfidf_ready:
        Models.ensure_tfidf(texts)

    # Threaded embedding
    pool_size = int(getattr(cfg, "EMBED_THREAD_POOL_SIZE", 4)) if cfg is not None else 4
    executor = Models.ensure_embed_executor(max_workers=pool_size)

    futures = []
    results: List[Optional[List[float]]] = [None] * len(texts)
    for idx, txt in enumerate(texts):
        futures.append((idx, executor.submit(_embed_single, model, txt)))

    for idx, fut in futures:
        try:
            emb = fut.result(timeout=getattr(cfg, "EMBED_TIMEOUT_SEC", 30))
            results[idx] = emb
            MET_EMBEDDINGS_PROCESSED.inc()
        except Exception:
            logger.exception("Embedding failed for index=%s", idx)
            results[idx] = None
            MET_EMBED_FAIL.inc()
    return results


# -----------------------
# High-level API
# -----------------------
def build_event_struct(
    *,
    cfg: Any,
    raw_text: str,
    source: str = "user",
    timestamp: Optional[Any] = None,
    symbol_hint: Optional[str] = None,
    extra: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Convert raw text into structured event dict ready for db.save_event.

    Args:
        cfg: configuration object
        raw_text: incoming raw text
        source: origin identifier
        timestamp: datetime or string (parsed to UTC)
        symbol_hint: optional symbol
        extra: optional dict

    Returns:
        dict matching db.save_event expected fields
    """
    ts = _parse_datetime(timestamp) if timestamp is not None else datetime.utcnow().replace(tzinfo=timezone.utc)
    cleaned = _clean_text(raw_text)
    symbols = extract_symbols(cleaned)
    symbol = symbol_hint or (symbols[0] if symbols else None)
    # classification: try ML classifier if available
    event_type = classify_event_type_rule(cleaned)
    clf = Models.event_clf
    if clf is None and getattr(cfg, "EVENT_CLASSIFIER_PATH", None):
        Models.load_event_classifier(getattr(cfg, "EVENT_CLASSIFIER_PATH"))
        clf = Models.event_clf
    if clf is not None:
        try:
            # simple vectorization via TF-IDF fallback; requires TF-IDF fit earlier
            if Models.tfidf_ready:
                vec = Models.tfidf.transform([cleaned]).toarray()
                pred = clf.predict(vec)[0]
                event_type = str(pred)
            else:
                logger.debug("Event classifier present but TF-IDF not fitted; using rule-based")
        except Exception:
            logger.exception("Event classifier failed; using rule-based")

    sent = sentiment_score(cleaned)
    rel = compute_relevance(cleaned, symbol)
    # impact hint heuristic
    try:
        length_factor = min(2.0, max(0.5, len(cleaned.split()) / 10.0))
        impact_hint = float(sent * rel * length_factor)
    except Exception:
        impact_hint = None

    # embedding: small single call (not for batch)
    emb = None
    try:
        emb = embed_texts_parallel([cleaned], cfg)[0]
    except Exception:
        logger.exception("Embedding single text failed for build_event_struct")

    ev = {
        "symbol": symbol,
        "source": source,
        "event_type": event_type,
        "text": cleaned,
        "sentiment": float(sent),
        "relevance": float(rel),
        "impact_hint": float(impact_hint) if impact_hint is not None else None,
        "embedding": emb,
        "timestamp": ts,
        "extra": extra or {},
    }
    return ev


def process_and_persist_event(
    cfg: Any,
    raw_text: str,
    source: str = "user",
    timestamp: Optional[Any] = None,
    symbol_hint: Optional[str] = None,
    extra: Optional[dict] = None,
    enqueue_impact: bool = True,
) -> int:
    """
    Process raw text into event and persist. Non-blocking: if configured, impact calculation is enqueued.

    Args:
        cfg: config
        raw_text: text
        source: source identifier
        timestamp: timestamp
        symbol_hint: optional
        enqueue_impact: whether to enqueue background impact calc (default True)

    Returns:
        saved event id
    """
    ev = build_event_struct(cfg=cfg, raw_text=raw_text, source=source, timestamp=timestamp, symbol_hint=symbol_hint, extra=extra)
    eid = db.save_event(cfg, ev)
    MET_EVENTS_PROCESSED.inc()
    # enqueue background task if configured
    if enqueue_impact and getattr(cfg, "AUTO_CALCULATE_IMPACT", False):
        try:
            _IMPACT_QUEUE.put_nowait(eid)
            MET_IMPACT_ENQUEUED.inc()
            logger.debug("Enqueued impact calc for event id=%s", eid)
        except Exception:
            logger.exception("Failed to enqueue impact task for event id=%s", eid)
    return eid


def process_batch(
    cfg: Any,
    items: Iterable[Dict[str, Any]],
    source: str = "batch",
    timestamp_field: str = "timestamp",
    symbol_field: str = "symbol",
    limit: Optional[int] = None,
) -> List[int]:
    """
    Process iterable of items (each has 'text' and optional metadata) in a batch.
    Embeddings are computed in parallel to speed up large batches.

    Args:
        cfg: config
        items: iterable of dicts
        source: source name
        timestamp_field: key for timestamp
        symbol_field: key for symbol hint
        limit: max items to ingest (fallback to cfg.MAX_EVENTS_PER_ANALYSIS)

    Returns:
        list of saved event ids
    """
    if limit is None:
        limit = int(getattr(cfg, "MAX_EVENTS_PER_ANALYSIS", 200))
    texts: List[str] = []
    raw_list: List[Dict[str, Any]] = []
    for it in items:
        if len(texts) >= limit:
            break
        txt = (it.get("text") or it.get("title") or it.get("body") or "").strip()
        if not txt:
            continue
        texts.append(_clean_text(txt))
        raw_list.append(it)

    if not texts:
        return []

    # compute embeddings in parallel
    embeddings = embed_texts_parallel(texts, cfg)

    saved_ids: List[int] = []
    for idx, raw in enumerate(raw_list):
        ts_raw = raw.get(timestamp_field)
        ts = _parse_datetime(ts_raw) if ts_raw is not None else datetime.utcnow().replace(tzinfo=timezone.utc)
        sym_hint = raw.get(symbol_field)
        text = texts[idx]
        emb = embeddings[idx] if idx < len(embeddings) else None
        # classification/sentiment/relevance (cheap)
        evt_type = classify_event_type_rule(text)
        sent = sentiment_score(text)
        rel = compute_relevance(text, sym_hint)
        try:
            ev = {
                "symbol": sym_hint,
                "source": source,
                "event_type": evt_type,
                "text": text,
                "sentiment": float(sent),
                "relevance": float(rel),
                "impact_hint": None,
                "embedding": emb,
                "timestamp": ts,
                "extra": raw.get("extra", {}),
            }
            eid = db.save_event(cfg, ev)
            saved_ids.append(eid)
        except Exception:
            logger.exception("Failed to save event in batch idx=%s", idx)
    return saved_ids


# -----------------------
# Classifier training helper (offline)
# -----------------------
def train_event_classifier(cfg: Any, texts: Sequence[str], labels: Sequence[str], save_path: Optional[str] = None) -> Optional[Any]:
    """
    Train a lightweight sklearn LogisticRegression classifier on given texts/labels.
    Requires sklearn.

    Args:
        cfg: config for TF-IDF settings
        texts: list of raw texts
        labels: list of labels matching texts
        save_path: optional path to persist trained model via joblib

    Returns:
        trained classifier or None on failure
    """
    if not _HAS_SKLEARN:
        logger.error("sklearn not available; cannot train classifier")
        return None
    try:
        Models.ensure_tfidf(texts)
        vect = Models.tfidf
        X = vect.transform(texts)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, labels)
        Models.event_clf = clf
        logger.info("Trained event classifier on %d samples", len(texts))
        if save_path:
            import joblib  # type: ignore

            joblib.dump(clf, save_path)
            logger.info("Saved event classifier to %s", save_path)
        return clf
    except Exception:
        logger.exception("Failed to train event classifier")
        return None


# -----------------------
# Lifecycle: initialize / shutdown
# -----------------------
def initialize(cfg: Any) -> None:
    """
    Initialize optional models, TF-IDF, and start background worker.

    Should be called once at app startup.
    """
    logger.info("Initializing NLP/Event pipeline...")
    Models.load_embedding_model(getattr(cfg, "EMBED_MODEL_NAME", _DEFAULT_EMBED_MODEL))
    Models.load_vader()
    # Pre-fit TF-IDF on any provided corpus to improve fallback (cfg.TFIDF_PRELOAD_TEXTS can be list)
    corpus = getattr(cfg, "TFIDF_PRELOAD_TEXTS", None)
    if corpus and _HAS_SKLEARN:
        try:
            Models.ensure_tfidf(corpus)
        except Exception:
            logger.exception("TF-IDF preload failed")
    # Start background worker for event impact if AUTO_CALCULATE_IMPACT enabled
    if getattr(cfg, "AUTO_CALCULATE_IMPACT", False):
        _start_impact_worker(cfg)
    logger.info("NLP/Event pipeline initialized")


def shutdown() -> None:
    """
    Shutdown background worker and executors. Call on app termination.
    """
    logger.info("Shutting down NLP/Event pipeline...")
    try:
        _stop_impact_worker()
    except Exception:
        logger.exception("Failed to stop impact worker")
    # Shutdown executor
    try:
        if Models._embed_executor:
            Models._embed_executor.shutdown(wait=True)
    except Exception:
        logger.exception("Failed to shutdown embed executor")
    logger.info("NLP/Event pipeline shutdown complete")