# main.py
"""
Entry point for quasihedge-bot.

Responsibilities:
- Load and validate configuration (app/config.py -> Config)
- Initialize logging (rotating file + console)
- Initialize DB (app.db) if available
- Initialize core bot orchestration (app.core) if available
- Start background scheduler / ingestion loops
- Graceful shutdown on SIGINT / SIGTERM
- Provide simple CLI: run | init-db | seed-demo | check-config

Place this file in the project root and run:

    python main.py run

The main file is intentionally defensive: if some modules (core, db) are not
yet implemented, it will log warnings but still allow running config checks
or DB initialization commands where possible.
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import logging.handlers
import signal
import sys
import traceback
from pathlib import Path
from typing import Optional

# Import the project's Config dataclass
try:
    from app.config import Config
except Exception as e:
    # If config is missing, we cannot proceed normally.
    print("FATAL: failed to import app.config.Config:", e)
    raise

# Try to import optional modules; main is tolerant if they are not present yet.
try:
    import app.db as db_module  # expected to implement init_db(), seed_demo()
except Exception:  # pragma: no cover - defensive
    db_module = None

try:
    import app.core as core_module  # expected to implement Bot/Orchestrator
except Exception:  # pragma: no cover - defensive
    core_module = None


def setup_logging(cfg: Config) -> None:
    """
    Configure root logger: console + rotating file handler.
    """
    log_path = Path(cfg.LOG_FILE)
    log_dir = log_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch_formatter = logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s")
    ch.setFormatter(ch_formatter)

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        filename=str(log_path),
        maxBytes=cfg.LOG_ROTATION_MB * 1024 * 1024,
        backupCount=cfg.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh_formatter = logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s:%(lineno)d] %(message)s")
    fh.setFormatter(fh_formatter)

    # Avoid duplicate handlers on reload
    if root_logger.handlers:
        root_logger.handlers = []

    root_logger.addHandler(ch)
    root_logger.addHandler(fh)

    logging.getLogger("urllib3").setLevel(logging.WARNING)  # quieter
    logging.getLogger("asyncio").setLevel(logging.WARNING)


async def _run_orchestrator(cfg: Config, stop_event: asyncio.Event) -> None:
    """
    Initialize and run the core orchestrator (if available).
    This coroutine returns when the bot stops (stop_event is set or orchestrator ends).
    """
    logger = logging.getLogger("main.orch")
    if core_module is None:
        logger.warning("app.core module not found. Core orchestrator will not run.")
        # Wait until stop event is set to allow long-running container to stay alive
        await stop_event.wait()
        return

    # Preferred interface: core_module.Orchestrator(cfg) with async start/stop methods
    orchestrator = None
    try:
        # Two possible expected patterns:
        # 1) core_module.Orchestrator(cfg) -> instance with start()/stop()/run_forever()
        # 2) core_module.run(cfg, stop_event) -> coroutine entrypoint
        if hasattr(core_module, "Orchestrator"):
            orchestrator = core_module.Orchestrator(cfg)
            logger.info("Starting Orchestrator...")
            # assume async start/stop
            if hasattr(orchestrator, "start") and asyncio.iscoroutinefunction(orchestrator.start):
                await orchestrator.start()
            elif hasattr(orchestrator, "run_forever") and asyncio.iscoroutinefunction(orchestrator.run_forever):
                # run_forever until stop_event set
                run_task = asyncio.create_task(orchestrator.run_forever(stop_event))
                await stop_event.wait()
                run_task.cancel()
            else:
                # fallback synchronous start
                start = getattr(orchestrator, "start", None)
                if callable(start):
                    start()
                await stop_event.wait()

            # graceful stop
            if hasattr(orchestrator, "stop") and asyncio.iscoroutinefunction(orchestrator.stop):
                await orchestrator.stop()
            elif hasattr(orchestrator, "stop") and callable(orchestrator.stop):
                orchestrator.stop()

        elif hasattr(core_module, "run") and asyncio.iscoroutinefunction(core_module.run):
            # core_module.run(cfg, stop_event) is a coroutine that should return when finished
            logger.info("Starting core_module.run coroutine...")
            await core_module.run(cfg, stop_event)
        else:
            logger.warning("app.core does not expose expected entrypoints (Orchestrator/run). No-op.")
            await stop_event.wait()
    except asyncio.CancelledError:
        logger.info("Orchestrator task cancelled.")
    except Exception:
        logger.exception("Uncaught exception in orchestrator:")
    finally:
        # final cleanup if orchestrator exists
        if orchestrator is not None:
            try:
                if hasattr(orchestrator, "stop") and callable(orchestrator.stop):
                    maybe = orchestrator.stop()
                    if asyncio.iscoroutine(maybe):
                        await maybe
            except Exception:
                logger.exception("Error during orchestrator.stop() cleanup.")


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    logger = logging.getLogger("main.signals")

    def _signal_handler(sig):
        logger.info("Received signal %s. Triggering shutdown.", sig.name)
        stop_event.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _signal_handler, s)
        except NotImplementedError:
            # Windows fallback: signal handlers added via signal.signal only for main thread
            signal.signal(s, lambda sig, frame: stop_event.set())


async def _main_run(cfg: Config) -> int:
    """
    Core async runner: initialize DB, orchestrator and wait for shutdown.
    """
    logger = logging.getLogger("main")
    stop_event = asyncio.Event()

    # Install handlers for SIGINT/SIGTERM to set stop_event
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, stop_event)

    # DB initialization (if module exists)
    if db_module is not None and hasattr(db_module, "init_db"):
        try:
            logger.info("Initializing database...")
            maybe = db_module.init_db(cfg)  # can be sync or async
            if asyncio.iscoroutine(maybe):
                await maybe
            logger.info("Database initialized.")
        except Exception:
            logger.exception("Failed to initialize DB.")

    # Start orchestrator
    orch_task = asyncio.create_task(_run_orchestrator(cfg, stop_event))

    # Wait until stop event is set
    try:
        await stop_event.wait()
        logging.getLogger("main").info("Shutdown requested, waiting for orchestrator to finish...")
        # Give orchestrator some time to finish
        await asyncio.wait_for(orch_task, timeout=30.0)
    except asyncio.TimeoutError:
        logging.getLogger("main").warning("Orchestrator did not exit within timeout, cancelling task...")
        orch_task.cancel()
        try:
            await orch_task
        except Exception:
            pass
    except Exception:
        logging.getLogger("main").exception("Unexpected exception in main run loop:")
    finally:
        logging.getLogger("main").info("Main run finished, exiting.")
    return 0


def run_sync(cfg: Config) -> int:
    """
    Synchronous wrapper to run async main runner.
    """
    logger = logging.getLogger("main")
    # Use uvloop if available (optional performance)
    try:
        import uvloop  # type: ignore

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.debug("uvloop enabled.")
    except Exception:
        logger.debug("uvloop not available or not desired; using default loop.")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_main_run(cfg))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down.")
        return 0
    except Exception:
        logger.exception("Unhandled exception in run_sync:")
        return 2
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


def cmd_init_db(cfg: Config) -> int:
    logger = logging.getLogger("main.initdb")
    if db_module is None:
        logger.error("DB module not found (app.db). Cannot init DB.")
        return 1
    if not hasattr(db_module, "init_db"):
        logger.error("app.db.init_db not found.")
        return 1
    try:
        logger.info("Initializing database via app.db.init_db() ...")
        res = db_module.init_db(cfg)
        if asyncio.iscoroutine(res):
            asyncio.run(res)
        logger.info("Database initialization finished.")
        return 0
    except Exception:
        logger.exception("Failed to initialize database.")
        return 2


def cmd_seed_demo(cfg: Config) -> int:
    logger = logging.getLogger("main.seed")
    if db_module is None:
        logger.error("DB module not found (app.db). Cannot seed demo data.")
        return 1
    if not hasattr(db_module, "seed_demo"):
        logger.error("app.db.seed_demo not found.")
        return 1
    try:
        logger.info("Seeding demo data via app.db.seed_demo() ...")
        res = db_module.seed_demo(cfg)
        if asyncio.iscoroutine(res):
            asyncio.run(res)
        logger.info("Demo data seeded.")
        return 0
    except Exception:
        logger.exception("Failed to seed demo data.")
        return 2


def cmd_check_config(cfg: Config) -> int:
    logger = logging.getLogger("main.check")
    logger.info("Configuration preview (sensitive values masked):")
    cfg.debug_print()
    # Basic connectivity warnings
    if cfg.ALPHAVANTAGE_ENABLED and not cfg.ALPHAVANTAGE_API_KEY:
        logger.warning("ALPHAVANTAGE_ENABLED but no key is set.")
    if cfg.ENABLE_TELEGRAM_INGEST and not cfg.TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM ingestion enabled but TELEGRAM_BOT_TOKEN not set.")
    if not cfg.DATA_DIR.exists():
        logger.warning("DATA_DIR does not exist: %s", cfg.DATA_DIR)
    return 0


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quasihedge-bot", description="QuasiHedge Bot - medium-quant assistant")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="Run the bot (start orchestrator and background tasks).")
    sub.add_parser("init-db", help="Initialize the database schema (calls app.db.init_db).")
    sub.add_parser("seed-demo", help="Seed demo data into DB (calls app.db.seed_demo).")
    sub.add_parser("check-config", help="Validate and print configuration.")

    return p


def main() -> int:
    # Load config first
    cfg = Config.load()

    # Setup logging
    setup_logging(cfg)
    logger = logging.getLogger("main")
    logger.info("Starting quasihedge-bot main; env=%s debug=%s", cfg.APP_ENV, cfg.DEBUG)

    # Parse CLI
    parser = build_cli()
    args = parser.parse_args()

    cmd = args.cmd
    if cmd == "run":
        logger.info("Running in 'run' mode.")
        return run_sync(cfg)
    elif cmd == "init-db":
        return cmd_init_db(cfg)
    elif cmd == "seed-demo":
        return cmd_seed_demo(cfg)
    elif cmd == "check-config":
        return cmd_check_config(cfg)
    else:
        logger.error("Unknown command: %s", cmd)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # If something catastrophic happens while starting, print traceback so user can debug.
        print("Fatal error starting main:")
        traceback.print_exc()
        sys.exit(10)