# app/config.py
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()  # загружает .env из корня проекта (если есть)


def _str2bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    v_lower = v.strip().lower()
    if v_lower in ("1", "true", "yes", "y", "on"):
        return True
    if v_lower in ("0", "false", "no", "n", "off"):
        return False
    return default


def _int(v: Optional[str], default: int) -> int:
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def _float(v: Optional[str], default: float) -> float:
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def _path(v: Optional[str], default: str) -> Path:
    return Path(v) if v else Path(default)


def _mask_secret(s: Optional[str]) -> str:
    if not s:
        return "<empty>"
    if len(s) <= 8:
        return s[:2] + "***"
    return s[:4] + ("*" * (len(s) - 8)) + s[-4:]


@dataclass
class Config:
    # CORE
    APP_ENV: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    DEBUG: bool = field(default_factory=lambda: _str2bool(os.getenv("DEBUG"), True))
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    DATA_DIR: Path = field(default_factory=lambda: _path(os.getenv("DATA_DIR"), "./data"))
    MODELS_DIR: Path = field(default_factory=lambda: _path(os.getenv("MODELS_DIR"), "./models"))
    LOGS_DIR: Path = field(default_factory=lambda: _path(os.getenv("LOGS_DIR"), "./logs"))

    # TELEGRAM
    TELEGRAM_BOT_TOKEN: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN"))
    TELEGRAM_API_ID: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_API_ID"))
    TELEGRAM_API_HASH: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_API_HASH"))
    ENABLE_TELEGRAM_INGEST: bool = field(default_factory=lambda: _str2bool(os.getenv("ENABLE_TELEGRAM_INGEST"), True))
    ENABLE_TELEGRAM_COMMANDS: bool = field(default_factory=lambda: _str2bool(os.getenv("ENABLE_TELEGRAM_COMMANDS"), True))

    # FIN DATA APIS
    YAHOO_ENABLED: bool = field(default_factory=lambda: _str2bool(os.getenv("YAHOO_ENABLED"), True))
    ALPHAVANTAGE_API_KEY: Optional[str] = field(default_factory=lambda: os.getenv("ALPHAVANTAGE_API_KEY"))
    ALPHAVANTAGE_ENABLED: bool = field(default_factory=lambda: _str2bool(os.getenv("ALPHAVANTAGE_ENABLED"), True))
    COINGECKO_API_KEY: Optional[str] = field(default_factory=lambda: os.getenv("COINGECKO_API_KEY"))
    COINGECKO_ENABLED: bool = field(default_factory=lambda: _str2bool(os.getenv("COINGECKO_ENABLED"), True))
    SEC_ENABLED: bool = field(default_factory=lambda: _str2bool(os.getenv("SEC_ENABLED"), True))

    # RSS / News
    ENABLE_RSS_INGEST: bool = field(default_factory=lambda: _str2bool(os.getenv("ENABLE_RSS_INGEST"), True))
    RSS_FETCH_INTERVAL_MINUTES: int = field(default_factory=lambda: _int(os.getenv("RSS_FETCH_INTERVAL_MINUTES"), 30))
    FINVIZ_ENABLED: bool = field(default_factory=lambda: _str2bool(os.getenv("FINVIZ_ENABLED"), True))
    SEC_RSS_ENABLED: bool = field(default_factory=lambda: _str2bool(os.getenv("SEC_RSS_ENABLED"), True))
    SEC_RSS_URL: str = field(default_factory=lambda: os.getenv("SEC_RSS_URL", "https://www.sec.gov/Archives/edgar/usgaap.rss"))

    # EVENT ENGINE
    MIN_ANALOGS_REQUIRED: int = field(default_factory=lambda: _int(os.getenv("MIN_ANALOGS_REQUIRED"), 30))
    DEDUP_COSINE_THRESHOLD: float = field(default_factory=lambda: _float(os.getenv("DEDUP_COSINE_THRESHOLD"), 0.85))
    EVENT_HALF_LIFE_DAYS: int = field(default_factory=lambda: _int(os.getenv("EVENT_HALF_LIFE_DAYS"), 30))
    MAX_EVENTS_PER_ANALYSIS: int = field(default_factory=lambda: _int(os.getenv("MAX_EVENTS_PER_ANALYSIS"), 200))
    ENABLE_EVENT_ML: bool = field(default_factory=lambda: _str2bool(os.getenv("ENABLE_EVENT_ML"), True))
    EVENT_MODEL_VERSION: str = field(default_factory=lambda: os.getenv("EVENT_MODEL_VERSION", "v1"))
    EVENT_CONFIDENCE_THRESHOLD: float = field(default_factory=lambda: _float(os.getenv("EVENT_CONFIDENCE_THRESHOLD"), 0.4))

    # SCORING
    WEIGHT_FUNDAMENTAL: float = field(default_factory=lambda: _float(os.getenv("WEIGHT_FUNDAMENTAL"), 0.35))
    WEIGHT_MARKET: float = field(default_factory=lambda: _float(os.getenv("WEIGHT_MARKET"), 0.30))
    WEIGHT_EVENT: float = field(default_factory=lambda: _float(os.getenv("WEIGHT_EVENT"), 0.35))
    BASE_BENCHMARK: str = field(default_factory=lambda: os.getenv("BASE_BENCHMARK", "SPY"))
    CRYPTO_BENCHMARK: str = field(default_factory=lambda: os.getenv("CRYPTO_BENCHMARK", "BTC"))

    # MONTE CARLO
    ENABLE_MONTE_CARLO: bool = field(default_factory=lambda: _str2bool(os.getenv("ENABLE_MONTE_CARLO"), True))
    MC_DEFAULT_N_SIMS: int = field(default_factory=lambda: _int(os.getenv("MC_DEFAULT_N_SIMS"), 2000))
    MC_DEFAULT_HORIZON_DAYS: int = field(default_factory=lambda: _int(os.getenv("MC_DEFAULT_HORIZON_DAYS"), 365))
    ENABLE_BOOTSTRAP_MC: bool = field(default_factory=lambda: _str2bool(os.getenv("ENABLE_BOOTSTRAP_MC"), True))
    ENABLE_REGIME_SWITCHING_MC: bool = field(default_factory=lambda: _str2bool(os.getenv("ENABLE_REGIME_SWITCHING_MC"), False))
    MC_TARGET_RETURN: float = field(default_factory=lambda: _float(os.getenv("MC_TARGET_RETURN"), 0.20))
    MC_DRAWDOWN_ALERT: float = field(default_factory=lambda: _float(os.getenv("MC_DRAWDOWN_ALERT"), 0.30))

    # RISK MANAGEMENT
    BASE_POSITION_SIZE: float = field(default_factory=lambda: _float(os.getenv("BASE_POSITION_SIZE"), 0.05))
    MAX_SINGLE_POSITION: float = field(default_factory=lambda: _float(os.getenv("MAX_SINGLE_POSITION"), 0.20))
    MAX_PORTFOLIO_DRAWDOWN: float = field(default_factory=lambda: _float(os.getenv("MAX_PORTFOLIO_DRAWDOWN"), 0.30))
    TARGET_PORTFOLIO_VOL: float = field(default_factory=lambda: _float(os.getenv("TARGET_PORTFOLIO_VOL"), 0.08))
    CONFIDENCE_STRONG: float = field(default_factory=lambda: _float(os.getenv("CONFIDENCE_STRONG"), 0.7))
    CONFIDENCE_MEDIUM: float = field(default_factory=lambda: _float(os.getenv("CONFIDENCE_MEDIUM"), 0.5))

    # DATABASE / LOGGING
    DATABASE_PATH: Path = field(default_factory=lambda: _path(os.getenv("DATABASE_PATH"), "./data/store.sqlite"))
    DB_ECHO: bool = field(default_factory=lambda: _str2bool(os.getenv("DB_ECHO"), False))
    LOG_FILE: Path = field(default_factory=lambda: _path(os.getenv("LOG_FILE"), "./logs/bot.log"))
    LOG_ROTATION_MB: int = field(default_factory=lambda: _int(os.getenv("LOG_ROTATION_MB"), 10))
    LOG_BACKUP_COUNT: int = field(default_factory=lambda: _int(os.getenv("LOG_BACKUP_COUNT"), 5))

    # REGIME DETECTION
    REGIME_VOL_LOOKBACK_DAYS: int = field(default_factory=lambda: _int(os.getenv("REGIME_VOL_LOOKBACK_DAYS"), 252))
    REGIME_TREND_FAST: int = field(default_factory=lambda: _int(os.getenv("REGIME_TREND_FAST"), 50))
    REGIME_TREND_SLOW: int = field(default_factory=lambda: _int(os.getenv("REGIME_TREND_SLOW"), 200))

    # derived / metadata
    raw_env: Dict[str, Any] = field(default_factory=lambda: dict(os.environ), repr=False)

    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        cfg._validate_and_fix()
        return cfg

    def _validate_and_fix(self) -> None:
        # create directories if needed
        for p in (self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR, self.LOG_FILE.parent, self.DATABASE_PATH.parent):
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        # sanity: scoring weights sum
        total_w = self.WEIGHT_FUNDAMENTAL + self.WEIGHT_MARKET + self.WEIGHT_EVENT
        if abs(total_w - 1.0) > 1e-6:
            # нормализуем, но логируем предупреждение (print чтоб не требовать логера ещё)
            print(f"[config] WARNING: scoring weights sum to {total_w:.4f}, normalizing to 1.0")
            self.WEIGHT_FUNDAMENTAL /= total_w
            self.WEIGHT_MARKET /= total_w
            self.WEIGHT_EVENT /= total_w

        # minimum thresholds sanity
        self.DEDUP_COSINE_THRESHOLD = max(0.5, min(0.99, self.DEDUP_COSINE_THRESHOLD))
        self.EVENT_CONFIDENCE_THRESHOLD = max(0.0, min(1.0, self.EVENT_CONFIDENCE_THRESHOLD))

        # basic recommendations: if no API keys but enabled => warn
        if self.ALPHAVANTAGE_ENABLED and not self.ALPHAVANTAGE_API_KEY:
            print("[config] WARNING: ALPHAVANTAGE_ENABLED=True but ALPHAVANTAGE_API_KEY is empty.")

        if self.ENABLE_TELEGRAM_INGEST and not self.TELEGRAM_BOT_TOKEN:
            print("[config] WARNING: ENABLE_TELEGRAM_INGEST=True but TELEGRAM_BOT_TOKEN is empty.")

        # clamp some ratios
        self.MAX_SINGLE_POSITION = max(0.0, min(1.0, self.MAX_SINGLE_POSITION))
        self.BASE_POSITION_SIZE = max(0.0, min(1.0, self.BASE_POSITION_SIZE))
        self.TARGET_PORTFOLIO_VOL = max(0.0, self.TARGET_PORTFOLIO_VOL)

    def as_dict(self, mask_secrets: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        # remove raw_env for readability
        d.pop("raw_env", None)
        if mask_secrets:
            for k in ("TELEGRAM_BOT_TOKEN", "ALPHAVANTAGE_API_KEY", "COINGECKO_API_KEY", "TELEGRAM_API_HASH"):
                if k in d:
                    d[k] = _mask_secret(d[k])
        # resolve Path -> str
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    def debug_print(self) -> None:
        d = self.as_dict(mask_secrets=True)
        print("=== CONFIG (masked secrets) ===")
        for k, v in d.items():
            print(f"{k} = {v}")
        print("=== end config ===")