# app/__init__.py
"""
QuasiHedge "app" package initializer.

Responsibilities:
- Provide convenient app-level helpers to bootstrap the system:
    * load_config(...) — load configuration from environment (and .env when available)
    * setup_logging(cfg) — configure logging (console + optional rotating file)
    * create_orchestrator(cfg) — create orchestrator instance (from app.core)
    * run(cfg=None) — start the app (blocking), gracefully handle shutdown
    * get_orchestrator() — return the running orchestrator (if created)
- Export commonly used symbols for external scripts/tests
- Be defensive: if some optional submodules are missing (app.config), provide a lightweight fallback
- Keep side-effects minimal on import (no network calls, no background threads started)

Notes:
- This module is intentionally compact but production-ready as an entrypoint for CLI/containers.
- It plays nicely with the core/orchestrator implemented in app.core.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import types
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

# Try to get package version if available
try:
    # Python 3.8+: importlib.metadata
    from importlib.metadata import version as _pkg_version  # type: ignore

    try:
        __version__ = _pkg_version("quasihedge")  # package name if installed
    except Exception:
        __version__ = "0.0.0"
except Exception:
    __version__ = "0.0.0"

# Attempt to import real Config (if user created app.config). If not available, use fallback.
try:
    from .config import Config  # type: ignore
    _HAS_REAL_CONFIG = True
except Exception:
    _HAS_REAL_CONFIG = False
    # Define a minimal, environment-driven Config shim used when app.config is not present.
    @dataclass
    class Config:
        """
        Lightweight Config shim which maps common env vars into attributes.
        It's intentionally permissive: code should use getattr(cfg, 'KEY', default).
        """
        # Core toggles
        APP_ENV: str = field(default_factory=lambda: os.environ.get("APP_ENV", "development"))
        DEBUG: bool = field(default_factory=lambda: os.environ.get("DEBUG", "False").lower() in ("1", "true", "yes"))
        LOG_LEVEL: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
        LOG_FILE: Optional[str] = field(default_factory=lambda: os.environ.get("LOG_FILE", None) or os.environ.get("LOGS_DIR", "./logs") + "/bot.log")

        # DB / dirs
        DATA_DIR: str = field(default_factory=lambda: os.environ.get("DATA_DIR", "./data"))
        MODELS_DIR: str = field(default_factory=lambda: os.environ.get("MODELS_DIR", "./models"))
        DATABASE_URL: Optional[str] = field(default_factory=lambda: os.environ.get("DATABASE_URL", os.environ.get("DATABASE_PATH", "./data/store.sqlite")))

        # Telegram
        TELEGRAM_BOT_TOKEN: Optional[str] = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN"))
        ENABLE_TELEGRAM_COMMANDS: bool = field(default_factory=lambda: os.environ.get("ENABLE_TELEGRAM_COMMANDS", "True").lower() in ("1", "true", "yes"))

        # Data API keys
        ALPHAVANTAGE_API_KEY: Optional[str] = field(default_factory=lambda: os.environ.get("ALPHAVANTAGE_API_KEY"))
        COINGECKO_API_KEY: Optional[str] = field(default_factory=lambda: os.environ.get("COINGECKO_API_KEY"))

        # MC / modeling defaults (can be overridden via env)
        MC_DEFAULT_N_SIMS: int = field(default_factory=lambda: int(os.environ.get("MC_DEFAULT_N_SIMS", "2000")))
        MC_DEFAULT_HORIZON_DAYS: int = field(default_factory=lambda: int(os.environ.get("MC_DEFAULT_HORIZON_DAYS", "365")))
        ENABLE_MONTE_CARLO: bool = field(default_factory=lambda: os.environ.get("ENABLE_MONTE_CARLO", "True").lower() in ("1", "true", "yes"))

        # Other defaults from .env examples
        BASE_BENCHMARK: str = field(default_factory=lambda: os.environ.get("BASE_BENCHMARK", "SPY"))
        PERIODS_PER_YEAR: int = field(default_factory=lambda: int(os.environ.get("PERIODS_PER_YEAR", "252")))
        MIN_ANALOGS_REQUIRED: int = field(default_factory=lambda: int(os.environ.get("MIN_ANALOGS_REQUIRED", "30")))
        EVENT_HALF_LIFE_DAYS: int = field(default_factory=lambda: int(os.environ.get("EVENT_HALF_LIFE_DAYS", "30")))

        # Generic map of env -> attribute (allows getattr(cfg, 'FOO'))
        extras: Dict[str, Any] = field(default_factory=dict)

        def __post_init__(self) -> None:
            # copy any remaining env vars into extras for convenience
            for k, v in os.environ.items():
                if not hasattr(self, k):
                    self.extras[k] = v

        def to_dict(self) -> Dict[str, Any]:
            """Return a plain dict representation (useful for logging)."""
            d = {k: getattr(self, k) for k in ("APP_ENV", "DEBUG", "LOG_LEVEL", "DATA_DIR", "MODELS_DIR", "TELEGRAM_BOT_TOKEN")}
            d.update(self.extras)
            return d


# Keep module-level orchestrator instance (created by create_orchestrator / run)
_orchestrator: Optional[Any] = None


# -----------------------
# Logging setup helper
# -----------------------
def setup_logging(cfg: Optional[Config] = None, *, console_level: Optional[str] = None) -> None:
    """
    Configure root logging for the package.

    Uses cfg.LOG_LEVEL / cfg.LOG_FILE when available. Safe to call multiple times.

    Args:
        cfg: configuration object (optional). Must provide LOG_LEVEL and LOG_FILE ideally.
        console_level: override for console handler level (string like 'INFO').
    """
    log_level = "INFO"
    log_file = None
    if cfg is not None:
        try:
            log_level = getattr(cfg, "LOG_LEVEL", log_level) or log_level
            log_file = getattr(cfg, "LOG_FILE", None)
        except Exception:
            pass
    if console_level:
        log_level = console_level

    level = getattr(logging, str(log_level).upper(), logging.INFO)
    root = logging.getLogger()
    # If root already configured with handlers, we update level and return
    if root.handlers:
        root.setLevel(level)
        for h in root.handlers:
            h.setLevel(level)
        return

    root.setLevel(level)
    fmt = "%(asctime)s %(levelname)5s [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(console)

    # Rotating file if configured
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
            root.addHandler(file_handler)
        except Exception:
            # fallback: log but continue with console only
            root.warning("Failed to create log file handler for %s; continuing without file logging", log_file)


# -----------------------
# Config loader
# -----------------------
def load_config(env_path: Optional[str] = None) -> Config:
    """
    Load configuration object.

    Behavior:
    - If package has app.config.Config (real), instantiate and return it.
      That class can implement its own validation / parsing.
    - Otherwise, return the lightweight shim Config which reads environment variables.
    - If env_path provided and python-dotenv is installed, load it.
    """
    # optionally load .env
    if env_path and os.path.exists(env_path):
        try:
            # lazy import python-dotenv (optional)
            from dotenv import load_dotenv  # type: ignore

            load_dotenv(env_path)
        except Exception:
            # ignore missing dependency
            pass

    # If user provided a real Config implementation, prefer it
    if _HAS_REAL_CONFIG:
        try:
            # The real Config class may expect to be instantiated without args or accept a path
            try:
                return Config()
            except TypeError:
                # try passing env_path
                return Config(env_path)
        except Exception:
            # fallback to shim below
            logging.getLogger(__name__).exception("app.config.Config failed to instantiate; falling back to shim Config")

    # fallback shim
    return Config()


# -----------------------
# Orchestrator helpers
# -----------------------
def create_orchestrator(cfg: Optional[Config] = None) -> Any:
    """
    Create an Orchestrator instance (does NOT start it).

    Requires app.core.Orchestrator to be present.

    Returns:
        orchestrator instance
    """
    global _orchestrator
    if cfg is None:
        cfg = load_config()
    try:
        from .core import Orchestrator  # type: ignore

        orchestrator = Orchestrator(cfg)
        _orchestrator = orchestrator
        return orchestrator
    except Exception as exc:  # pragma: no cover - defensive
        logging.getLogger(__name__).exception("Failed to import Orchestrator from app.core: %s", exc)
        raise


async def _async_start_and_wait(orchestrator: Any) -> None:
    """
    Start orchestrator and wait until cancelled (internal helper).
    """
    # start all subsystems
    await orchestrator.start()
    # Simple wait loop; will exit when a CancelledError is raised from outside (run -> stop)
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        # shutdown will be performed by caller
        raise


def run(cfg: Optional[Config] = None, *, use_signals: bool = True) -> None:
    """
    Blocking run entrypoint.

    - Loads config if needed
    - Sets up logging
    - Starts orchestrator and blocks until KeyboardInterrupt or SIGTERM
    - On shutdown, attempts graceful orchestrator.stop()

    Example:
        from app import run
        run()

    Returns nothing; exits when terminated.
    """
    global _orchestrator
    if cfg is None:
        cfg = load_config()
    setup_logging(cfg)

    log = logging.getLogger(__name__)
    log.info("Starting QuasiHedge app (version %s)...", __version__)
    # Instantiate orchestrator
    from .core import Orchestrator  # type: ignore

    orch = Orchestrator(cfg)
    _orchestrator = orch

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Helper to handle signals gracefully by canceling the main task
    stop_event = asyncio.Event()

    def _signal_handler(signame: str) -> None:
        log.info("Received signal %s, shutting down...", signame)
        loop.call_soon_threadsafe(stop_event.set)

    if use_signals:
        try:
            loop.add_signal_handler(signal.SIGINT, lambda: _signal_handler("SIGINT"))
            loop.add_signal_handler(signal.SIGTERM, lambda: _signal_handler("SIGTERM"))
        except Exception:
            # add_signal_handler may not be available on Windows for ProactorEventLoop
            pass

    async def _main():
        try:
            await orch.start()
            log.info("Orchestrator started. Waiting for shutdown signal...")
            # wait until stop_event is set
            await stop_event.wait()
        finally:
            # ensure stop called
            try:
                await orch.stop()
            except Exception:
                log.exception("Error while stopping orchestrator")

    try:
        loop.run_until_complete(_main())
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received, shutting down")
        try:
            loop.run_until_complete(orch.stop())
        except Exception:
            log.exception("Failed to stop orchestrator after KeyboardInterrupt")
    finally:
        # Close loop and cleanup
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()
        log.info("QuasiHedge app stopped.")


def get_orchestrator() -> Optional[Any]:
    """Return the orchestrator instance if created, else None."""
    return _orchestrator


# Exports
__all__ = [
    "Config",
    "load_config",
    "setup_logging",
    "create_orchestrator",
    "run",
    "get_orchestrator",
    "__version__",
]