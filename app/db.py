# app/db.py
"""
Database module for quasihedge-bot.

Production-ready SQLAlchemy 2.0 style storage layer supporting:
- events, price_history, event_impact, channels, watchlist, model_registry
- JSON embeddings
- indices and composite indices
- FK relations with cascade delete
- bulk upsert for PostgreSQL (ON CONFLICT) and SQLite (UPSERT)
- analogs engine with normalized cosine similarity
- event study engine with robust p-value handling (scipy optional)
- type hints and robust logging
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy import create_engine
from sqlalchemy.sql import func

# Optional scipy import for statistics
try:
    from scipy import stats as scipy_stats  # type: ignore

    _HAS_SCIPY = True
except Exception:
    scipy_stats = None  # type: ignore
    _HAS_SCIPY = False

# Logging
logger = logging.getLogger("app.db")


# -----------------------
# Base / ORM models
# -----------------------
class Base(DeclarativeBase):
    """SQLAlchemy declarative base for ORM models."""


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(32), index=True, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    event_type: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    sentiment: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    relevance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    impact_hint: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    embedding: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # stored as JSON list/dict
    dedup_hash: Mapped[Optional[str]] = mapped_column(String(128), index=True, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    processed: Mapped[str] = mapped_column(String(5), nullable=False, default="false")
    extra: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    impacts = relationship(
        "EventImpact",
        back_populates="event",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_events_symbol_type_time", "symbol", "event_type", "timestamp"),
    )


class PriceHistory(Base):
    __tablename__ = "price_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    open: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    adj_close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uix_symbol_timestamp"),
        Index("ix_price_symbol_timestamp", "symbol", "timestamp"),
    )


class EventImpact(Base):
    __tablename__ = "event_impact"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(Integer, ForeignKey("events.id", ondelete="CASCADE"), nullable=False, index=True)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    excess_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    event = relationship("Event", back_populates="impacts")


class Channel(Base):
    __tablename__ = "channels"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)  # 'rss' or 'telegram'
    identifier: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)  # url or tg id
    enabled: Mapped[bool] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class Watchlist(Base):
    __tablename__ = "watchlist"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    meta_info: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    explain: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    mc_report: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    input_events: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    extra: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


def save_analysis_result(cfg: Any, ar: Dict[str, Any]) -> int:
    """
    Save analysis result dictionary into analyses table.
    """
    session = get_session()
    try:
        a = Analysis(
            ticker=ar.get("ticker") or ar.get("symbol") or "<unknown>",
            score=ar.get("score"),
            confidence=ar.get("confidence"),
            explain=ar.get("explain"),
            mc_report=ar.get("mc_report"),
            model_version=ar.get("model_version"),
            input_events=ar.get("input_events"),
            extra=ar.get("extra"),
            created_at=datetime.utcnow(),
        )
        session.add(a)
        session.commit()
        session.refresh(a)
        logger.info("Saved analysis id=%s for %s", a.id, a.ticker)
        return a.id
    finally:
        session.close()


def get_recent_analyses(cfg: Any, ticker: str, limit: int = 5) -> List[Analysis]:
    session = get_session()
    try:
        stmt = select(Analysis).where(Analysis.ticker == ticker).order_by(Analysis.created_at.desc()).limit(limit)
        return session.scalars(stmt).all()
    finally:
        session.close()


# -----------------------
# Engine / session management
# -----------------------
SessionLocal = None  # type: ignore
_engine: Optional[Engine] = None


def _cfg_to_database_url(cfg: Any) -> str:
    # prefer explicit DATABASE_URL, else derive from DATABASE_PATH (sqlite)
    db_url = getattr(cfg, "DATABASE_URL", None)
    if db_url:
        return db_url
    db_path = getattr(cfg, "DATABASE_PATH", None)
    if db_path:
        p = Path(db_path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        abs_path = str(p.resolve())
        return f"sqlite:///{abs_path}"
    return "sqlite:///./data/store.sqlite"


def init_engine(database_url: str | None = None, echo: bool = False, cfg: Any | None = None) -> Engine:
    """
    Initialize and return SQLAlchemy engine and session factory.

    If database_url is None and cfg provided, derive URL from cfg.
    """
    global _engine, SessionLocal
    if _engine is None:
        if database_url is None and cfg is not None:
            database_url = _cfg_to_database_url(cfg)
        if database_url is None:
            database_url = "sqlite:///./data/store.sqlite"
        _engine = create_engine(database_url, future=True, echo=echo)
        SessionLocal = sessionmaker(bind=_engine, future=True, expire_on_commit=False)
        logger.info("Engine initialized for %s", database_url.split("://", 1)[0])
    return _engine


def init_db(cfg: Any) -> None:
    """
    Create DB schema if not exists.

    Args:
        cfg: configuration object providing DATABASE_URL or DATABASE_PATH
    """
    database_url = _cfg_to_database_url(cfg)
    engine = init_engine(database_url, echo=getattr(cfg, "DB_ECHO", False))
    Base.metadata.create_all(engine)
    logger.info("Database schema created/checked at %s", database_url)


def get_session():
    """
    Get a new Session instance.

    Returns:
        Session
    """
    global SessionLocal
    if SessionLocal is None:
        raise RuntimeError("Engine/session not initialized. Call init_engine() or init_db() first.")
    return SessionLocal()


# -----------------------
# Utility helpers
# -----------------------
def _make_uid(source: str, timestamp: datetime, text: str) -> str:
    """Generate deterministic uid for event."""
    import hashlib

    base = f"{source}|{timestamp.isoformat()}|{text[:512]}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _safe_normalize(vec: Sequence[float]) -> Optional[np.ndarray]:
    """
    Return L2-normalized numpy vector or None if invalid.

    Args:
        vec: sequence of floats

    Returns:
        normalized numpy array or None
    """
    if vec is None:
        return None
    try:
        arr = np.asarray(vec, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0:
        return None
    norm = np.linalg.norm(arr)
    if norm == 0 or np.isnan(norm):
        return None
    return arr / norm


# -----------------------
# CRUD: Events
# -----------------------
def save_event(cfg: Any, event: Dict[str, Any], dedup_threshold: float | None = None) -> int:
    """
    Save event with deduplication by hash or embedding similarity.

    Args:
        cfg: config (needed for ANALOG_LIMIT etc.)
        event: dict with keys: symbol, source, event_type, text, sentiment, relevance,
               impact_hint, embedding (list), timestamp (datetime), extra
        dedup_threshold: override threshold; default from cfg.DEDUP_COSINE_THRESHOLD

    Returns:
        event.id
    """
    dedup_threshold = dedup_threshold if dedup_threshold is not None else getattr(cfg, "DEDUP_COSINE_THRESHOLD", 0.85)
    session = get_session()
    try:
        text = event.get("text", "").strip()
        if not text:
            logger.warning("Attempt to save empty event text; skipping.")
            raise ValueError("Event text empty.")

        source = event.get("source", "unknown")
        timestamp = event.get("timestamp", datetime.utcnow())
        uid = event.get("uid") or _make_uid(source, timestamp, text)
        dedup_hash = event.get("dedup_hash")
        if dedup_hash is None:
            import hashlib

            dedup_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Exact dedup by uid or hash
        existing = session.scalar(select(Event).where(Event.uid == uid))
        if existing:
            logger.debug("Exact UID dedup found for uid=%s", uid)
            return existing.id
        existing = session.scalar(select(Event).where(Event.dedup_hash == dedup_hash))
        if existing:
            logger.debug("Exact hash dedup found for hash=%s", dedup_hash)
            return existing.id

        embedding = event.get("embedding")
        normalized = _safe_normalize(embedding) if embedding is not None else None
        # Embedding-based dedup: compare to recent events for same symbol (configurable limit)
        analog_limit = int(getattr(cfg, "ANALOG_LIMIT", 500))
        if normalized is not None:
            stmt = select(Event).where(Event.symbol == event.get("symbol")).order_by(Event.timestamp.desc()).limit(analog_limit)
            candidates = session.scalars(stmt).all()
            for c in candidates:
                if c.embedding:
                    try:
                        cand_emb = np.asarray(c.embedding, dtype=np.float64)
                        cand_norm = cand_emb / np.linalg.norm(cand_emb) if np.linalg.norm(cand_emb) != 0 else None
                        if cand_norm is not None:
                            sim = float(np.dot(normalized, cand_norm))
                            if sim >= dedup_threshold:
                                logger.debug("Embedding dedup: sim=%.4f with event id=%s", sim, c.id)
                                return c.id
                    except Exception:
                        # skip broken embeddings
                        continue

        ev = Event(
            uid=uid,
            symbol=event.get("symbol"),
            source=source,
            event_type=event.get("event_type"),
            text=text,
            sentiment=event.get("sentiment"),
            relevance=event.get("relevance"),
            impact_hint=event.get("impact_hint"),
            embedding=list(normalized.tolist()) if normalized is not None else (event.get("embedding") or None),
            dedup_hash=dedup_hash,
            timestamp=timestamp,
            created_at=datetime.utcnow(),
            processed="false",
            extra=event.get("extra"),
        )
        session.add(ev)
        session.commit()
        logger.info("Saved event id=%s symbol=%s type=%s", ev.id, ev.symbol, ev.event_type)
        return ev.id
    except IntegrityError:
        session.rollback()
        # try to fetch by uid
        existing = session.scalar(select(Event).where(Event.uid == uid))
        if existing:
            logger.warning("IntegrityError while saving event; returning existing id=%s", existing.id)
            return existing.id
        raise
    finally:
        session.close()


def get_event_by_id(cfg: Any, event_id: int) -> Optional[Event]:
    """Return Event by id."""
    session = get_session()
    try:
        return session.get(Event, event_id)
    finally:
        session.close()


def list_events(cfg: Any, symbol: Optional[str] = None, ticker: Optional[str] = None, limit: int = 200) -> List[Event]:
    """List recent events, optional filter by symbol (alias ticker supported)."""
    if ticker is not None:
        symbol = ticker
    session = get_session()
    try:
        stmt = select(Event).order_by(Event.timestamp.desc())
        if symbol:
            stmt = stmt.where(Event.symbol == symbol)
        stmt = stmt.limit(limit)
        return session.scalars(stmt).all()
    finally:
        session.close()


# -----------------------
# Price history upsert (bulk) - optimized
# -----------------------
def save_price_history(cfg: Any, symbol: str, df: pd.DataFrame) -> None:
    """
    Bulk upsert price history for a symbol.

    Args:
        cfg: config
        symbol: ticker/symbol
        df: DataFrame with columns ['timestamp','open','high','low','close','adj_close','volume']
    """
    if df is None or df.empty:
        logger.warning("save_price_history called with empty df for %s", symbol)
        return

    # Normalize dataframe
    data = df.copy()
    if "timestamp" not in data.columns:
        if "date" in data.columns:
            data = data.rename(columns={"date": "timestamp"})
        else:
            raise ValueError("DataFrame must contain 'timestamp' or 'date' column.")

    data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.tz_localize(None)
    insert_rows: List[Dict[str, Any]] = []
    for _, row in data.iterrows():
        insert_rows.append(
            {
                "symbol": symbol,
                "timestamp": row["timestamp"].to_pydatetime(),
                "open": None if pd.isna(row.get("open")) else float(row.get("open")),
                "high": None if pd.isna(row.get("high")) else float(row.get("high")),
                "low": None if pd.isna(row.get("low")) else float(row.get("low")),
                "close": None if pd.isna(row.get("close")) else float(row.get("close")),
                "adj_close": None if pd.isna(row.get("adj_close")) else float(row.get("adj_close")),
                "volume": None if pd.isna(row.get("volume")) else float(row.get("volume")),
            }
        )

    engine = _engine
    if engine is None:
        engine = init_engine(None, echo=getattr(cfg, "DB_ECHO", False), cfg=cfg)

    session = get_session()
    try:
        dialect_name = engine.dialect.name.lower()
        table = PriceHistory.__table__
        if dialect_name.startswith("postgres"):
            stmt = pg_insert(table).values(insert_rows)
            update_dict = {c.name: stmt.excluded[c.name] for c in table.c if c.name not in ("id",)}
            stmt = stmt.on_conflict_do_update(index_elements=["symbol", "timestamp"], set_=update_dict)
            session.execute(stmt)
            session.commit()
            logger.info("Upserted %d rows (postgres) for %s", len(insert_rows), symbol)
        else:
            # SQLite or other: use sqlite insert with ON CONFLICT DO UPDATE (supported in modern SQLAlchemy)
            try:
                stmt = sqlite_insert(table).values(insert_rows)
                update_dict = {c.name: stmt.excluded[c.name] for c in table.c if c.name not in ("id",)}
                stmt = stmt.on_conflict_do_update(index_elements=["symbol", "timestamp"], set_=update_dict)
                session.execute(stmt)
                session.commit()
                logger.info("Upserted %d rows (sqlite) for %s", len(insert_rows), symbol)
            except Exception:
                # fallback: naive per-row upsert (slower)
                logger.warning("Dialect upsert unsupported; falling back to row-by-row upsert for %s", symbol)
                for r in insert_rows:
                    try:
                        existing = session.scalar(select(PriceHistory).where(PriceHistory.symbol == r["symbol"], PriceHistory.timestamp == r["timestamp"]))
                        if existing:
                            for k, v in r.items():
                                if k in ("symbol", "timestamp"):
                                    continue
                                setattr(existing, k, v)
                        else:
                            ph = PriceHistory(**r)
                            session.add(ph)
                    except Exception:
                        session.rollback()
                        logger.exception("Failed upsert row for %s at %s", r["symbol"], r["timestamp"])
                session.commit()
                logger.info("Upsert finished with fallback for %s", symbol)
    finally:
        session.close()


def get_price_history(cfg: Any, symbol: str, start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
    """
    Retrieve price history for a symbol as pandas DataFrame.

    Args:
        cfg: config
        symbol: ticker/symbol
        start: inclusive
        end: inclusive

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, adj_close, volume]
    """
    session = get_session()
    try:
        stmt = select(PriceHistory).where(PriceHistory.symbol == symbol)
        if start:
            stmt = stmt.where(PriceHistory.timestamp >= start)
        if end:
            stmt = stmt.where(PriceHistory.timestamp <= end)
        stmt = stmt.order_by(PriceHistory.timestamp.asc())
        rows = session.scalars(stmt).all()
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "adj_close", "volume"])
        data = []
        for r in rows:
            data.append(
                {
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "adj_close": r.adj_close,
                    "volume": r.volume,
                }
            )
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    finally:
        session.close()


# -----------------------
# Analogs Engine
# -----------------------
def find_analogs(
    cfg: Any,
    reference_event: Dict[str, Any],
    symbol: str,
    type_filter: Optional[str] = None,
    lookback_days: int = 3650,
    cosine_threshold: float = 0.75,
    limit: Optional[int] = None,
) -> Tuple[List[Event], Dict[str, Any]]:
    """
    Find historical analog events similar to reference_event.

    Args:
        cfg: config object (used for ANALOG_LIMIT fallback)
        reference_event: dict with keys 'event_type','embedding','timestamp','source',...
        symbol: symbol for which to search analogs
        type_filter: optional event type filter
        lookback_days: how far to look back in days
        cosine_threshold: similarity threshold to consider
        limit: max number of candidates to consider (None -> cfg.ANALOG_LIMIT or 500)

    Returns:
        (analogs_list, stats_dict)
    """
    limit = int(limit or getattr(cfg, "ANALOG_LIMIT", 500))
    ref_type = reference_event.get("event_type")
    ref_emb = reference_event.get("embedding")
    ref_ts = reference_event.get("timestamp", datetime.utcnow())
    cutoff = ref_ts - timedelta(days=lookback_days)
    session = get_session()
    try:
        # Candidate query: filter by time window and (optionally) type or symbol
        stmt = select(Event).where(Event.timestamp >= cutoff)
        if type_filter or ref_type:
            stmt = stmt.where(Event.event_type == (type_filter or ref_type))
        # Order by recency; we will compute similarity in Python numpy
        stmt = stmt.order_by(Event.timestamp.desc()).limit(limit * 3)  # fetch some extra for safety
        candidates = session.scalars(stmt).all()
        logger.debug("Fetched %d candidates for analog search (limit=%d)", len(candidates), limit)

        norm_ref = _safe_normalize(ref_emb) if ref_emb is not None else None

        analogs: List[Event] = []
        sims: List[float] = []

        for c in candidates:
            if c.symbol != symbol:
                continue  # prefer same-symbol events for analogs
            if norm_ref is None:
                # if no reference embedding, use exact type match
                if ref_type and c.event_type == ref_type:
                    analogs.append(c)
                continue
            if not c.embedding:
                continue
            cand_arr = _safe_normalize(c.embedding)
            if cand_arr is None:
                continue
            sim = float(np.dot(norm_ref, cand_arr))
            if sim >= cosine_threshold:
                analogs.append(c)
                sims.append(sim)
                logger.debug("Analog found id=%s sim=%.4f type=%s ts=%s", c.id, sim, c.event_type, c.timestamp)
            if len(analogs) >= limit:
                break

        stats: Dict[str, Any] = {"n_candidates": len(candidates), "n_analogs": len(analogs)}
        return analogs, stats
    finally:
        session.close()


# -----------------------
# Event Study Engine
# -----------------------
def compute_cumulative_return_from_df(df: pd.DataFrame, horizon_days: int) -> Optional[float]:
    """
    Compute cumulative return from first row to date >= first_date + horizon_days using adj_close.
    Returns None if insufficient data.
    """
    if df is None or df.empty or "adj_close" not in df.columns:
        return None
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    if df_sorted.shape[0] < 2:
        return None
    start_price = df_sorted.iloc[0]["adj_close"]
    if start_price is None or pd.isna(start_price) or start_price == 0:
        return None
    horizon_date = df_sorted.iloc[0]["timestamp"] + pd.Timedelta(days=horizon_days)
    later = df_sorted[df_sorted["timestamp"] >= horizon_date]
    if later.empty:
        end_price = df_sorted.iloc[-1]["adj_close"]
    else:
        end_price = later.iloc[0]["adj_close"]
    if end_price is None or pd.isna(end_price):
        return None
    try:
        return float(end_price / start_price - 1.0)
    except Exception:
        return None


def calculate_event_impact(cfg: Any, event_id: int) -> Optional[EventImpact]:
    """
    Calculate event impact metrics using historical analogs and store results in event_impact.

    Steps:
    - Fetch event by id
    - Find analogs (same type or similar embedding)
    - For configured horizons compute excess returns (symbol - benchmark)
    - Compute mean excess and p-value (if scipy available)
    - Store per-horizon records into event_impact table

    Args:
        cfg: config object (expects BASE_BENCHMARK, EVENT_WINDOWS)
        event_id: id of event to analyze

    Returns:
        EventImpact record for primary horizon (first in EVENT_WINDOWS) or None on failure
    """
    session = get_session()
    try:
        ev = session.get(Event, event_id)
        if ev is None:
            logger.warning("calculate_event_impact: event id=%s not found", event_id)
            return None

        windows = getattr(cfg, "EVENT_WINDOWS", [30, 90, 180])
        base_benchmark = getattr(cfg, "BASE_BENCHMARK", None)
        if not base_benchmark:
            logger.warning("No BASE_BENCHMARK configured; aborting impact calculation.")
            return None

        # Build reference_event dict
        ref = {"event_type": ev.event_type, "embedding": ev.embedding, "timestamp": ev.timestamp, "source": ev.source}

        analogs, analogs_stats = find_analogs(cfg, ref, ev.symbol, lookback_days=3650, cosine_threshold=getattr(cfg, "ANALOG_COSINE", 0.75), limit=getattr(cfg, "ANALOG_LIMIT", 500))
        n_analogs = analogs_stats.get("n_analogs", 0)
        logger.info("Found %d analogs for event id=%s", n_analogs, event_id)

        # For each analog compute excess return over windows
        results_by_horizon: Dict[int, List[float]] = {h: [] for h in windows}
        for a in analogs:
            for h in windows:
                # get price history for analog symbol starting at analog.timestamp
                df_sym = get_price_history(cfg, a.symbol, start=a.timestamp, end=a.timestamp + timedelta(days=h + 10))
                df_bench = get_price_history(cfg, base_benchmark, start=a.timestamp, end=a.timestamp + timedelta(days=h + 10))
                r_sym = compute_cumulative_return_from_df(df_sym, h)
                r_bench = compute_cumulative_return_from_df(df_bench, h)
                if r_sym is None or r_bench is None:
                    continue
                results_by_horizon[h].append(float(r_sym - r_bench))

        primary_result: Optional[EventImpact] = None
        # Save results
        for h in windows:
            arr = np.array(results_by_horizon[h]) if results_by_horizon[h] else np.array([])
            n = int(arr.size)
            mean_excess = float(np.mean(arr)) if n > 0 else None
            pval = None
            if n >= 2 and _HAS_SCIPY:
                try:
                    tstat, pval = scipy_stats.ttest_1samp(arr, 0.0)
                    pval = float(pval)
                except Exception:
                    pval = None
            # create EventImpact record
            impact = EventImpact(
                event_id=ev.id,
                horizon_days=int(h),
                excess_return=mean_excess,
                p_value=pval,
                n_samples=n,
                created_at=datetime.utcnow(),
            )
            session.add(impact)
            # pick primary as first horizon
            if primary_result is None:
                primary_result = impact
        session.commit()
        if primary_result:
            logger.info("Saved event impact records for event id=%s", event_id)
        else:
            logger.info("No impact records saved for event id=%s (insufficient data)", event_id)
        return primary_result
    except Exception:
        session.rollback()
        logger.exception("Failed to calculate event impact for event id=%s", event_id)
        raise
    finally:
        session.close()


# -----------------------
# Model registry / watchlist / channels helpers
# -----------------------
def register_model(cfg: Any, name: str, version: str, path: str, metadata: Optional[dict] = None) -> int:
    """
    Register model artifact in model_registry.
    """
    session = get_session()
    try:
        m = ModelRegistry(name=name, version=version, path=path, metadata=metadata or {}, created_at=datetime.utcnow())
        session.add(m)
        session.commit()
        session.refresh(m)
        logger.info("Registered model %s:%s id=%s", name, version, m.id)
        return m.id
    finally:
        session.close()


def add_watchlist_entry(cfg: Any, name: str, symbol: str, note: Optional[str] = None) -> int:
    session = get_session()
    try:
        w = Watchlist(name=name, symbol=symbol, note=note, created_at=datetime.utcnow())
        session.add(w)
        session.commit()
        session.refresh(w)
        logger.info("Added watchlist %s -> %s", name, symbol)
        return w.id
    finally:
        session.close()


def add_watchlist(cfg: Any, name: str, ticker: str, note: Optional[str] = None) -> int:
    """
    Compatibility wrapper: core calls add_watchlist(cfg, name=..., ticker=...)
    """
    return add_watchlist_entry(cfg, name=name, symbol=ticker, note=note)


def list_watchlist(cfg: Any) -> List[Watchlist]:
    session = get_session()
    try:
        return session.scalars(select(Watchlist).order_by(Watchlist.created_at.desc())).all()
    finally:
        session.close()


def add_channel(cfg: Any, name: str, source: str, identifier: str, enabled: bool = True) -> int:
    session = get_session()
    try:
        ch = Channel(name=name, source=source, identifier=identifier, enabled=bool(enabled), created_at=datetime.utcnow())
        session.add(ch)
        session.commit()
        session.refresh(ch)
        logger.info("Added channel %s (%s)", name, source)
        return ch.id
    finally:
        session.close()


def list_channels(cfg: Any) -> List[Channel]:
    session = get_session()
    try:
        return session.scalars(select(Channel).order_by(Channel.created_at.desc())).all()
    finally:
        session.close()


# -----------------------
# Quick inspect / seed demo
# -----------------------
def quick_inspect(cfg: Any, symbol: str, n_days: int = 365) -> Tuple[pd.DataFrame, List[Event]]:
    """
    Return recent price DataFrame and recent events for quick inspection.
    """
    df = get_price_history(cfg, symbol, start=datetime.utcnow() - timedelta(days=n_days))
    evs = list_events(cfg, symbol=symbol, limit=100)
    return df, evs


def seed_demo(cfg: Any) -> None:
    """
    Seed demo data suitable for local development.
    Creates synthetic price series for DEMOA and the base benchmark and some events.
    """
    logger.info("Seeding demo data...")
    init_db(cfg)
    session = get_session()
    try:
        now = datetime.utcnow()
        dates = pd.date_range(end=now, periods=400, freq="D")
        # Synthetic symbol DEMOA
        prices_a = 100.0 * np.cumprod(1 + 0.0008 + 0.01 * np.random.randn(len(dates)))
        df_a = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices_a * (1 - 0.002),
                "high": prices_a * (1 + 0.002),
                "low": prices_a * (1 - 0.002),
                "close": prices_a,
                "adj_close": prices_a,
                "volume": np.random.randint(1000, 10000, size=len(dates)),
            }
        )
        save_price_history(cfg, "DEMOA", df_a)

        # Benchmark
        prices_b = 300.0 * np.cumprod(1 + 0.0003 + 0.008 * np.random.randn(len(dates)))
        df_b = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices_b * (1 - 0.002),
                "high": prices_b * (1 + 0.002),
                "low": prices_b * (1 - 0.002),
                "close": prices_b,
                "adj_close": prices_b,
                "volume": np.random.randint(10000, 50000, size=len(dates)),
            }
        )
        benchmark = getattr(cfg, "BASE_BENCHMARK", "SPY")
        save_price_history(cfg, benchmark, df_b)

        # Events for DEMOA
        for i in range(1, 8):
            event_time = now - timedelta(days=30 * i)
            ev = {
                "symbol": "DEMOA",
                "source": "demo-rss",
                "event_type": "contract" if i % 2 == 0 else "report",
                "text": f"Demo event #{i} for DEMOA",
                "sentiment": 0.2 if i % 2 == 0 else -0.1,
                "relevance": 0.9,
                "impact_hint": 0.2,
                "embedding": (np.random.randn(128) * 0.01).tolist(),
                "timestamp": event_time,
                "extra": {"demo_index": i},
            }
            save_event(cfg, ev)
        logger.info("Demo data seeded.")
    finally:
        session.close()