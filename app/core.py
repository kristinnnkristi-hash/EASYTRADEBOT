# app/core.py
"""
Core orchestrator and Telegram command handlers for QuasiHedge Bot.

Responsibilities:
- Initialize subsystems: DB, NLP, Services, Modeling, MonteCarlo
- Start/stop application lifecycle (async-friendly)
- Provide Telegram command handlers (async) and a thin CLI fallback
- Implement high-level command flows:
    * start_analysis TICKER
    * add_info TICKER (or batch)
    * update_analysis TICKER
    * step_summary TICKER
    * simulate TICKER HORIZON
    * compare_history TICKER
    * analyze_batch T1,T2,...
    * auto_watch TICKER on/off
    * portfolio_review
- Background tasks:
    * auto-watch: poll configured sources for watched tickers and trigger analyses
    * periodic ingest (RSS/Telegram) if enabled
- Notifications & reporting formatting
- Defensive: degrades when optional modules missing

Notes:
- Uses asyncio and python-telegram-bot v20+ if available; otherwise runs without Telegram.
- Expects modules: app.config.Config, app.db, app.services, app.nlp_events, app.modeling, app.mc_risk
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

# Submodules
from app.config import Config
import app.db as db
import app.services as services
import app.nlp_events as nlp
import app.modeling as modeling
import app.mc_risk as mc_risk

# Telegram (async) support
try:
    from telegram import Update, __version__ as _ptb_ver  # type: ignore
    from telegram.ext import (
        Application,
        ApplicationBuilder,
        CommandHandler,
        MessageHandler,
        ContextTypes,
        filters,
    )  # type: ignore

    _HAS_TELEGRAM = True
except Exception:
    Application = None  # type: ignore
    ApplicationBuilder = None  # type: ignore
    Update = None  # type: ignore
    ContextTypes = None  # type: ignore
    CommandHandler = None  # type: ignore
    MessageHandler = None  # type: ignore
    filters = None  # type: ignore
    _HAS_TELEGRAM = False

logger = logging.getLogger("app.core")
logger.addHandler(logging.NullHandler())


@dataclass
class Orchestrator:
    cfg: Config
    telegram_app: Optional[Any] = None
    _tasks: Optional[List[asyncio.Task]] = None
    _watch_tasks: Optional[Dict[str, asyncio.Task]] = None
    _running: bool = False

    def __post_init__(self) -> None:
        self._tasks = [] if self._tasks is None else self._tasks
        self._watch_tasks = {} if self._watch_tasks is None else self._watch_tasks
        self.admin_chat_id = getattr(self.cfg, "ADMIN_CHAT_ID", None)
        self.watch_interval = int(getattr(self.cfg, "AUTO_WATCH_INTERVAL_SEC", 300))
        self.rss_interval = int(getattr(self.cfg, "RSS_FETCH_INTERVAL_MINUTES", 30)) * 60
        self._tg_poll_task: Optional[asyncio.Task] = None
        self._event_model = None

    # -----------------------
    # Lifecycle
    # -----------------------
    async def start(self) -> None:
        """Initialize DB, NLP, services, modeling and start background tasks."""
        logger.info("Orchestrator starting...")
        # init DB
        try:
            db.init_db(self.cfg)
            logger.info("DB initialized.")
        except Exception:
            logger.exception("DB initialization failed.")
            raise

        # initialize services (best-effort)
        try:
            if hasattr(services, "initialize"):
                await maybe_await(services.initialize(self.cfg))
            logger.info("Services initialized.")
        except Exception:
            logger.exception("Services initialization failed (continuing).")

        # initialize NLP pipeline
        try:
            nlp.initialize(self.cfg)
            logger.info("NLP pipeline initialized.")
        except Exception:
            logger.exception("NLP initialization failed (continuing).")

        # Prepare modeling artifacts if required
        try:
            model_path = getattr(self.cfg, "EVENT_REACTION_MODEL_PATH", None)
            if model_path:
                self._event_model = modeling.load_event_reaction_model(model_path)
                logger.info("Loaded event reaction model from %s", model_path)
            else:
                self._event_model = None
        except Exception:
            logger.exception("Loading event reaction model failed.")
            self._event_model = None

        # start background tasks
        if getattr(self.cfg, "ENABLE_RSS_INGEST", False) or getattr(self.cfg, "ENABLE_TELEGRAM_INGEST", False):
            t = asyncio.create_task(self._periodic_ingest_loop(), name="periodic_ingest")
            self._tasks.append(t)

        # resume watchlist tasks
        try:
            entries = db.list_watchlist(self.cfg)
            for e in entries:
                sym = getattr(e, "symbol", None)
                if sym and getattr(self.cfg, "AUTO_WATCH_ON_START", False):
                    await self._start_watch_for_symbol(sym)
        except Exception:
            logger.exception("Failed to initialize watchlist tasks")

        # start Telegram if token present and library available
        if _HAS_TELEGRAM and getattr(self.cfg, "ENABLE_TELEGRAM_COMMANDS", True) and getattr(self.cfg, "TELEGRAM_BOT_TOKEN", None):
            try:
                await self._start_telegram()
            except Exception:
                logger.exception("Failed to start Telegram application")
        else:
            logger.info("Telegram not started (missing token or library).")

        self._running = True
        logger.info("Orchestrator started.")

    async def stop(self) -> None:
        """Stop all background tasks, shutdown telegram and NLP pipelines."""
        logger.info("Orchestrator stopping...")
        self._running = False

        # cancel tasks
        for t in list(self._tasks):
            try:
                t.cancel()
            except Exception:
                logger.exception("Failed to cancel task %s", t)

        # cancel watch tasks
        for sym, task in list(self._watch_tasks.items()):
            try:
                task.cancel()
            except Exception:
                logger.exception("Failed to cancel watch task %s", task)
        self._watch_tasks.clear()

        # cancel telegram poll task if running
        if self._tg_poll_task:
            try:
                self._tg_poll_task.cancel()
                # give event loop a moment to process cancellation
                await asyncio.sleep(0)
                logger.info("Telegram polling task cancelled.")
            except Exception:
                logger.exception("Failed to cancel telegram poll task")

        # shutdown telegram app
        if self.telegram_app:
            try:
                # Attempt graceful stop/shutdown using available APIs
                if hasattr(self.telegram_app, "stop"):
                    await maybe_await(self.telegram_app.stop())
                if hasattr(self.telegram_app, "shutdown"):
                    await maybe_await(self.telegram_app.shutdown())
                logger.info("Telegram app stopped.")
            except Exception:
                logger.exception("Failed stopping telegram app.")

        # shutdown NLP resources
        try:
            nlp.shutdown()
        except Exception:
            logger.exception("NLP shutdown failed.")

        # stop services (if implemented)
        try:
            if hasattr(services, "shutdown"):
                await maybe_await(services.shutdown())
        except Exception:
            logger.exception("Services shutdown failed.")

        logger.info("Orchestrator stopped.")

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        """
        Run until stop_event is set.
        Orchestrator expects start() already called.
        """
        logger.info("Orchestrator running until stop_event")
        try:
            await stop_event.wait()
            logger.info("Stop event received")
        finally:
            await self.stop()

    # -----------------------
    # Telegram integration
    # -----------------------
    async def _start_telegram(self) -> None:
        """Initialize telegram Application and handlers, then start polling."""
        token = getattr(self.cfg, "TELEGRAM_BOT_TOKEN", None)
        if not token:
            logger.warning("Telegram token not configured; skipping telegram startup.")
            return
        if not _HAS_TELEGRAM:
            logger.warning("python-telegram-bot library not installed; skipping telegram.")
            return

        # Build application via builder (PTB v20+)
        try:
            self.telegram_app = ApplicationBuilder().token(token).build()
        except Exception:
            # Fallback to Application if builder missing
            try:
                if Application is not None:
                    self.telegram_app = Application()  # type: ignore
                else:
                    self.telegram_app = None
            except Exception:
                self.telegram_app = None

        if not self.telegram_app:
            logger.error("Failed to create Telegram Application instance.")
            return

        # Register command handlers
        try:
            self.telegram_app.add_handler(CommandHandler("start", self._cmd_start))
            self.telegram_app.add_handler(CommandHandler("start_analysis", self._cmd_start_analysis))
            self.telegram_app.add_handler(CommandHandler("add_info", self._cmd_add_info))
            self.telegram_app.add_handler(CommandHandler("update_analysis", self._cmd_update_analysis))
            self.telegram_app.add_handler(CommandHandler("step_summary", self._cmd_step_summary))
            self.telegram_app.add_handler(CommandHandler("simulate", self._cmd_simulate))
            self.telegram_app.add_handler(CommandHandler("compare_history", self._cmd_compare_history))
            self.telegram_app.add_handler(CommandHandler("analyze_batch", self._cmd_analyze_batch))
            self.telegram_app.add_handler(CommandHandler("auto_watch", self._cmd_auto_watch))
            self.telegram_app.add_handler(CommandHandler("portfolio_review", self._cmd_portfolio_review))
            self.telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._cmd_message))
        except Exception:
            logger.exception("Failed to register Telegram handlers (continuing).")

# Start application and polling safely
try:
    try:
        await self.telegram_app.initialize()
        await self.telegram_app.start()
        await self.telegram_app.updater.start_polling()
        logger.info("Telegram bot started and polling via async task.")
    except TypeError:
        loop = asyncio.get_running_loop()
        self._tg_poll_task = loop.create_task(
            loop.run_in_executor(None, self.telegram_app.run_polling)
        )
        logger.info("Telegram bot started and polling via executor.")

except Exception:
    logger.exception("Failed to fully start Telegram application.")

    # -----------------------
    # Command implementations (Telegram / programmatic)
    # -----------------------
    async def _cmd_start(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        "Simple welcome message"
        chat_id = get_chat_id(update)
        text = "QuasiHedge Bot ready. Use /start_analysis <TICKER> to begin."
        await safe_send_message(self.telegram_app, chat_id, text)

    async def _cmd_start_analysis(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """Handler: /start_analysis TICKER"""
        chat_id = get_chat_id(update)
        args = context.args if hasattr(context, "args") else []
        if not args:
            await safe_send_message(self.telegram_app, chat_id, "Usage: /start_analysis <TICKER>")
            return
        ticker = args[0].upper()
        await safe_send_message(self.telegram_app, chat_id, f"Starting analysis for {ticker} ...")
        # schedule background analysis task so we don't block handler
        asyncio.create_task(self._analyze_and_report(ticker, chat_id))

    async def _cmd_add_info(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """
        Handler: /add_info TICKER <then send a batch of messages or pipe a link>
        Inline usage supported: /add_info TICKER | text1; text2; text3
        """
        chat_id = get_chat_id(update)
        raw = " ".join(context.args) if hasattr(context, "args") else ""
        if "|" in raw:
            try:
                ticker, rest = raw.split("|", 1)
                ticker = ticker.strip().upper()
                parts = [p.strip() for p in re.split(r"[;\n\r]+", rest) if p.strip()]
                saved_ids = []
                for t in parts:
                    eid = nlp.process_and_persist_event(self.cfg, t, source=f"telegram_{chat_id}", symbol_hint=ticker)
                    saved_ids.append(eid)
                await safe_send_message(self.telegram_app, chat_id, f"Saved {len(saved_ids)} events for {ticker}.")
                return
            except Exception:
                logger.exception("Failed inline add_info parsing")
                await safe_send_message(self.telegram_app, chat_id, "Failed to parse /add_info inline format.")
                return
        await safe_send_message(self.telegram_app, chat_id, "Send messages with news/plans now; they will be saved for the ticker.")

    async def _cmd_update_analysis(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """Handler: /update_analysis TICKER"""
        chat_id = get_chat_id(update)
        args = context.args if hasattr(context, "args") else []
        if not args:
            await safe_send_message(self.telegram_app, chat_id, "Usage: /update_analysis <TICKER>")
            return
        ticker = args[0].upper()
        await safe_send_message(self.telegram_app, chat_id, f"Updating analysis for {ticker} ...")
        asyncio.create_task(self._analyze_and_report(ticker, chat_id, force_refresh=True))

    async def _cmd_step_summary(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """Handler: /step_summary TICKER"""
        chat_id = get_chat_id(update)
        args = context.args if hasattr(context, "args") else []
        if not args:
            await safe_send_message(self.telegram_app, chat_id, "Usage: /step_summary <TICKER>")
            return
        ticker = args[0].upper()
        await safe_send_message(self.telegram_app, chat_id, f"Preparing step summary for {ticker} ...")
        # _build_summary not implemented here — keep call but guard
        if hasattr(self, "_build_summary"):
            res = await maybe_await(self._build_summary(ticker))
            await safe_send_message(self.telegram_app, chat_id, res)
        else:
            await safe_send_message(self.telegram_app, chat_id, "Step summary not available.")

    async def _cmd_simulate(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """Handler: /simulate TICKER HORIZON_MONTHS"""
        chat_id = get_chat_id(update)
        args = context.args if hasattr(context, "args") else []
        if len(args) < 2:
            await safe_send_message(self.telegram_app, chat_id, "Usage: /simulate <TICKER> <HORIZON_MONTHS>")
            return
        ticker = args[0].upper()
        try:
            horizon = int(args[1])
        except Exception:
            await safe_send_message(self.telegram_app, chat_id, "HORIZON_MONTHS must be integer (1,6,12).")
            return
        await safe_send_message(self.telegram_app, chat_id, f"Running Monte-Carlo for {ticker} {horizon}M ...")
        asyncio.create_task(self._simulate_and_report(ticker, horizon, chat_id))

    async def _cmd_compare_history(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """Handler: /compare_history TICKER"""
        chat_id = get_chat_id(update)
        args = context.args if hasattr(context, "args") else []
        if not args:
            await safe_send_message(self.telegram_app, chat_id, "Usage: /compare_history <TICKER>")
            return
        ticker = args[0].upper()
        await safe_send_message(self.telegram_app, chat_id, f"Searching historical analogs for {ticker} ...")
        asyncio.create_task(self._compare_history_and_report(ticker, chat_id))

    async def _cmd_analyze_batch(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """Handler: /analyze_batch T1,T2,..."""
        chat_id = get_chat_id(update)
        args_raw = " ".join(context.args) if hasattr(context, "args") else ""
        if not args_raw:
            await safe_send_message(self.telegram_app, chat_id, "Usage: /analyze_batch T1,T2,...")
            return
        tickers = [t.strip().upper() for t in re.split(r"[,; ]+", args_raw) if t.strip()]
        await safe_send_message(self.telegram_app, chat_id, f"Starting batch analysis for {len(tickers)} tickers ...")
        asyncio.create_task(self._analyze_batch_and_report(tickers, chat_id))

    async def _cmd_auto_watch(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """Handler: /auto_watch TICKER on|off"""
        chat_id = get_chat_id(update)
        args = context.args if hasattr(context, "args") else []
        if len(args) < 2:
            await safe_send_message(self.telegram_app, chat_id, "Usage: /auto_watch <TICKER> <on|off>")
            return
        ticker = args[0].upper()
        flag = args[1].lower()
        if flag in ("on", "1", "true", "yes"):
            await self._enable_watch(ticker, chat_id)
            await safe_send_message(self.telegram_app, chat_id, f"Auto-watch enabled for {ticker}")
        else:
            await self._disable_watch(ticker, chat_id)
            await safe_send_message(self.telegram_app, chat_id, f"Auto-watch disabled for {ticker}")

    async def _cmd_portfolio_review(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """Handler: /portfolio_review"""
        chat_id = get_chat_id(update)
        await safe_send_message(self.telegram_app, chat_id, "Running portfolio review...")
        asyncio.create_task(self._portfolio_review_and_report(chat_id))

    async def _cmd_message(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """
        Generic text message handler. For now supports:
        - If message starts with "\analysis " treat the rest as a link or text batch and process
        - Otherwise, ignore (or remind user)
        """
        chat_id = get_chat_id(update)
        text = update.message.text.strip() if update and update.message and update.message.text else ""
        if text.lower().startswith("\\analysis "):
            payload = text[len("\\analysis ") :].strip()
            parts = [p.strip() for p in re.split(r"[;\n\r]+", payload) if p.strip()]
            if not parts:
                await safe_send_message(self.telegram_app, chat_id, "No data found in analysis payload.")
                return
            symbol_hint = None
            first = parts[0]
            if re.match(r"^[A-Z0-9]{1,6}$", first):
                symbol_hint = first.upper()
                parts = parts[1:]
            await safe_send_message(self.telegram_app, chat_id, f"Processing {len(parts)} supplied items for analysis...")
            ids = nlp.process_batch(self.cfg, [{"text": p, "symbol": symbol_hint} for p in parts], source=f"tg_{chat_id}")
            await safe_send_message(self.telegram_app, chat_id, f"Saved {len(ids)} events. You can now use /update_analysis {symbol_hint or '<ticker>'}.")
            return
        await safe_send_message(self.telegram_app, chat_id, "Message received. For batch analysis use \\analysis <TICKER?> <items...> or /help.")

    # -----------------------
    # High-level flows (background tasks)
    # -----------------------
    async def _analyze_and_report(self, ticker: str, chat_id: Optional[int] = None, force_refresh: bool = False) -> None:
        """
        Orchestrate data fetching, scoring, MC, risk and report generation.
        """
        logger.info("Analyze_and_report started for %s", ticker)
        try:
            # fetch price
            price_df = await maybe_await(services.fetch_price_history(self.cfg, ticker))
            if price_df is None or (hasattr(price_df, "empty") and price_df.empty):
                price_df = db.get_price_history(self.cfg, ticker)

            # fetch benchmark
            bench_symbol = getattr(self.cfg, "BASE_BENCHMARK", None)
            bench_df = None
            if bench_symbol:
                bench_df = await maybe_await(services.fetch_price_history(self.cfg, bench_symbol))
                if bench_df is None or (hasattr(bench_df, "empty") and bench_df.empty):
                    bench_df = db.get_price_history(self.cfg, bench_symbol)

            # fundamentals
            fundamentals = await maybe_await(services.fetch_fundamentals(self.cfg, ticker))

            # events
            events = [e_to_dict(ev) for ev in db.list_events(self.cfg, ticker=ticker, limit=200)]

            # modeling
            result = modeling.combine_scores(
                cfg=self.cfg,
                symbol=ticker,
                price_df=price_df,
                fundamentals=fundamentals,
                events=events,
                benchmark_df=bench_df,
                model=getattr(self, "_event_model", None),
            )

            # save analysis to DB
            try:
                ar = {
                    "ticker": ticker,
                    "score": float(result.get("final_score_0_1", 0.0)),
                    "confidence": float(result.get("confidence", 0.0)),
                    "explain": result.get("explain"),
                    "mc_report": None,
                    "model_version": getattr(self._event_model, "version", None) if getattr(self, "_event_model", None) else None,
                    "input_events": [e["uid"] for e in db.list_events(self.cfg, ticker=ticker, limit=200) if getattr(e, "uid", None)],
                    "extra": {"generated_at": datetime.utcnow().isoformat()},
                }
                db.save_analysis_result(self.cfg, ar)
            except Exception:
                logger.exception("Saving analysis result failed")

            # optionally run Monte-Carlo if configured (can be heavy)
            if getattr(self.cfg, "ENABLE_MONTE_CARLO", True):
                try:
                    n_sims = int(getattr(self.cfg, "MC_DEFAULT_N_SIMS", 2000))
                    horizon_days = int(getattr(self.cfg, "MC_DEFAULT_HORIZON_DAYS", 365))
                    mc = mc_risk.run_gbm_mc(self.cfg, price_df, horizon_days, n_sims, news_adjustments=events)
                    result["mc_report"] = mc
                except Exception:
                    logger.exception("Monte-Carlo simulation failed")
                    result["mc_report"] = None

            # build message
            msg = format_analysis_result(result)
            if chat_id:
                await safe_send_message(self.telegram_app, chat_id, msg)
            else:
                if self.admin_chat_id:
                    await safe_send_message(self.telegram_app, self.admin_chat_id, f"Analysis complete for {ticker}:\n{msg}")
            logger.info("Analysis complete for %s", ticker)
        except Exception:
            logger.exception("Analyze_and_report failed for %s", ticker)
            if chat_id:
                await safe_send_message(self.telegram_app, chat_id, f"Analysis failed for {ticker}. Check logs.")
            else:
                if self.admin_chat_id:
                    await safe_send_message(self.telegram_app, self.admin_chat_id, f"Analysis failed for {ticker} (see logs).")

    async def _simulate_and_report(self, ticker: str, horizon_months: int, chat_id: Optional[int]) -> None:
        """Run Monte-Carlo and report results."""
        try:
            price_df = await maybe_await(services.fetch_price_history(self.cfg, ticker))
            if price_df is None or (hasattr(price_df, "empty") and price_df.empty):
                price_df = db.get_price_history(self.cfg, ticker)
            horizon_days = int(horizon_months * 30)
            n_sims = int(getattr(self.cfg, "MC_DEFAULT_N_SIMS", 2000))
            mc = mc_risk.run_gbm_mc(self.cfg, price_df, horizon_days, n_sims)
            text = format_mc_report(mc)
            if chat_id:
                await safe_send_message(self.telegram_app, chat_id, text)
        except Exception:
            logger.exception("Simulation failed for %s", ticker)
            if chat_id:
                await safe_send_message(self.telegram_app, chat_id, f"Simulation failed for {ticker}.")

    async def _compare_history_and_report(self, ticker: str, chat_id: Optional[int]) -> None:
        """Find analogs and report summary stats."""
        try:
            events = db.list_events(self.cfg, ticker=ticker, limit=50)
            if not events:
                await safe_send_message(self.telegram_app, chat_id, f"No events saved for {ticker}.")
                return
            latest = events[0]
            ref = {"event_type": latest.event_type, "embedding": latest.embedding, "timestamp": latest.timestamp, "source": latest.source}
            analogs, stats = db.find_analogs(self.cfg, ref, ticker)
            txt = f"Found {stats.get('n_analogs', 0)} analogs for latest event of {ticker}.\n"
            for h in ("30", "90", "180"):
                mean_k = f"mean_excess_{h}"
                n_k = f"n_{h}"
                p_k = f"p_value_{h}"
                txt += f"- Horizon {h}d: n={stats.get(n_k,0)} mean_excess={stats.get(mean_k)} p={stats.get(p_k)}\n"
            await safe_send_message(self.telegram_app, chat_id, txt)
        except Exception:
            logger.exception("compare_history failed for %s", ticker)
            await safe_send_message(self.telegram_app, chat_id, "compare_history failed (see logs).")

    async def _analyze_batch_and_report(self, tickers: Sequence[str], chat_id: Optional[int]) -> None:
        """Analyze a list of tickers and send a concise report (top picks)."""
        try:
            results: Dict[str, Dict[str, float]] = {}
            for t in tickers:
                await self._analyze_and_report(t, None)
                last = db.get_recent_analyses(self.cfg, ticker=t, limit=1)
                if last:
                    a = last[0]
                    results[t] = {"score": getattr(a, "score", 0.0), "confidence": getattr(a, "confidence", 0.0)}
            lines = ["Batch analysis complete. Results:"]
            for t, r in results.items():
                lines.append(f"{t}: score={r['score']:.3f} conf={r['confidence']:.2f}")
            await safe_send_message(self.telegram_app, chat_id, "\n".join(lines))
        except Exception:
            logger.exception("analyze_batch failed")
            await safe_send_message(self.telegram_app, chat_id, "analyze_batch failed (see logs).")

    # -----------------------
    # Auto-watch management
    # -----------------------
    async def _enable_watch(self, ticker: str, chat_id: Optional[int] = None) -> None:
        """Enable auto-watch: add to watchlist and start task."""
        try:
            if hasattr(db, "add_watchlist"):
                db.add_watchlist(self.cfg, name=f"auto_{ticker}", ticker=ticker)
            elif hasattr(db, "add_watchlist_entry"):
                db.add_watchlist_entry(self.cfg, name=f"auto_{ticker}", symbol=ticker)
            else:
                logger.debug("No watchlist add API available in db module.")
        except Exception:
            logger.exception("Failed to add watchlist entry for %s", ticker)
        await self._start_watch_for_symbol(ticker)

    async def _disable_watch(self, ticker: str, chat_id: Optional[int] = None) -> None:
        """Disable auto-watch: cancel task and remove watchlist entries (all)."""
        task = self._watch_tasks.get(ticker)
        if task:
            try:
                task.cancel()
            except Exception:
                logger.exception("Failed to cancel watch task for %s", ticker)
            self._watch_tasks.pop(ticker, None)
        # remove watchlist rows (best-effort)
        try:
            session_list = db.list_watchlist(self.cfg)
            for w in session_list:
                if getattr(w, "symbol", None) == ticker:
                    if hasattr(db, "remove_watchlist"):
                        try:
                            db.remove_watchlist(self.cfg, getattr(w, "id", None))
                        except Exception:
                            logger.exception("Failed to remove watchlist id=%s", getattr(w, "id", None))
        except Exception:
            logger.exception("Failed to cleanup watchlist rows for %s", ticker)

    async def _start_watch_for_symbol(self, ticker: str) -> None:
        """Spawn a background task that periodically checks for new events/news for ticker and triggers analysis if significant."""
        if ticker in self._watch_tasks:
            logger.info("Watch already running for %s", ticker)
            return
        logger.info("Starting watch task for %s", ticker)
        task = asyncio.create_task(self._watch_loop(ticker), name=f"watch_{ticker}")
        self._watch_tasks[ticker] = task

    async def _watch_loop(self, ticker: str) -> None:
        """
        Polling loop:
        - fetch recent messages from services (rss/telegram)
        - run nlp.process_batch for new items
        - if new strong events appear, run update analysis and optionally notify
        """
        logger.info("Watch loop started for %s", ticker)
        last_seen = None
        try:
            while True:
                try:
                    raw_items = []
                    if hasattr(services, "fetch_recent_messages_for_symbol"):
                        try:
                            raw_items = await maybe_await(services.fetch_recent_messages_for_symbol(self.cfg, ticker, since=last_seen))
                        except Exception:
                            logger.exception("fetch_recent_messages_for_symbol failed")
                    else:
                        if getattr(self.cfg, "ENABLE_RSS_INGEST", False) and hasattr(services, "fetch_rss"):
                            try:
                                feeds = getattr(self.cfg, "MONITOR_RSS_FEEDS", [])
                                for f in feeds:
                                    items = await maybe_await(services.fetch_rss(self.cfg, f, since=last_seen))
                                    if items:
                                        raw_items.extend(items)
                            except Exception:
                                logger.exception("RSS fetch failed in watch loop")
                    # filter items mentioning ticker
                    new_items = []
                    for it in raw_items:
                        txt = (it.get("text") or it.get("title") or "") if isinstance(it, dict) else str(it)
                        if ticker.upper() in txt.upper():
                            new_items.append({"text": txt, "timestamp": it.get("timestamp") if isinstance(it, dict) else None, "extra": it})
                    if new_items:
                        ids = nlp.process_batch(self.cfg, [{"text": it["text"], "timestamp": it.get("timestamp"), "symbol": ticker} for it in new_items], source=f"watch_{ticker}")
                        logger.info("Watch found %d new items for %s", len(ids), ticker)
                        events = [e_to_dict(db.get_event_by_id(self.cfg, eid)) for eid in ids if db.get_event_by_id(self.cfg, eid)]
                        avg_sent = sum((ev.get("sentiment", 0) for ev in events), 0.0) / max(1, len(events))
                        avg_rel = sum((ev.get("relevance", 0) for ev in events), 0.0) / max(1, len(events))
                        logger.debug("Watch stats %s avg_sent=%.3f avg_rel=%.3f", ticker, avg_sent, avg_rel)
                        if avg_rel > 0.5 and abs(avg_sent) > 0.2:
                            await self._analyze_and_report(ticker, self.admin_chat_id)
                    last_seen = datetime.utcnow()
                except asyncio.CancelledError:
                    logger.info("Watch loop cancelled for %s", ticker)
                    raise
                except Exception:
                    logger.exception("Error in watch loop for %s", ticker)
                await asyncio.sleep(self.watch_interval)
        finally:
            logger.info("Watch loop exiting for %s", ticker)

    # -----------------------
    # Periodic ingest loop (RSS/Telegram)
    # -----------------------
    async def _periodic_ingest_loop(self) -> None:
        """Periodically poll configured RSS/Telegram sources and persist new items."""
        logger.info("Periodic ingest loop started")
        try:
            while True:
                try:
                    if getattr(self.cfg, "ENABLE_RSS_INGEST", False) and hasattr(services, "fetch_all_rss"):
                        try:
                            all_items = await maybe_await(services.fetch_all_rss(self.cfg))
                            logger.info("Fetched %d RSS items", len(all_items) if all_items else 0)
                            if all_items:
                                items = [{"text": it.get("title") or it.get("summary") or it.get("content") or "", "timestamp": it.get("published")} for it in all_items]
                                nlp.process_batch(self.cfg, items, source="rss_periodic")
                        except Exception:
                            logger.exception("Periodic RSS fetch failed")
                    if getattr(self.cfg, "ENABLE_TELEGRAM_INGEST", False) and hasattr(services, "fetch_all_telegram"):
                        try:
                            tg_items = await maybe_await(services.fetch_all_telegram(self.cfg))
                            if tg_items:
                                items = [{"text": it.get("text") or "", "timestamp": it.get("timestamp")} for it in tg_items]
                                nlp.process_batch(self.cfg, items, source="telegram_periodic")
                        except Exception:
                            logger.exception("Periodic Telegram fetch failed")
                except asyncio.CancelledError:
                    logger.info("Periodic ingest loop cancelled")
                    raise
                except Exception:
                    logger.exception("Periodic ingest loop error")
                await asyncio.sleep(self.rss_interval)
        finally:
            logger.info("Periodic ingest loop stopped")

    # -----------------------
    # Portfolio review
    # -----------------------
    async def _portfolio_review_and_report(self, chat_id: Optional[int]) -> None:
        """Run scoring for watchlist entries and produce a consolidated report."""
        try:
            entries = db.list_watchlist(self.cfg)
            symbols = [getattr(e, "symbol", None) for e in entries]
            symbols = [s for s in symbols if s]
            if not symbols:
                await safe_send_message(self.telegram_app, chat_id, "Watchlist is empty.")
                return
            resp_lines = []
            for s in symbols:
                price_df = await maybe_await(services.fetch_price_history(self.cfg, s))
                if price_df is None or (hasattr(price_df, "empty") and price_df.empty):
                    price_df = db.get_price_history(self.cfg, s)
                fund = await maybe_await(services.fetch_fundamentals(self.cfg, s))
                events = [e_to_dict(ev) for ev in db.list_events(self.cfg, ticker=s, limit=200)]
                res = modeling.combine_scores(cfg=self.cfg, symbol=s, price_df=price_df, fundamentals=fund, events=events, benchmark_df=None, model=getattr(self, "_event_model", None))
                score_10 = res.get("final_score_10", 0.0) if isinstance(res, dict) else 0.0
                conf = res.get("confidence", 0.0) if isinstance(res, dict) else 0.0
                pos_size = (res.get("position", {}).get("size", 0.0) if isinstance(res, dict) else 0.0)
                resp_lines.append(f"{s}: score={score_10:.1f}/10 conf={conf:.2f} pos={pos_size:.2%}")
            await safe_send_message(self.telegram_app, chat_id, "Portfolio review:\n" + "\n".join(resp_lines))
        except Exception:
            logger.exception("Portfolio review failed")
            await safe_send_message(self.telegram_app, chat_id, "Portfolio review failed (see logs).")


# -----------------------
# Helper utilities
# -----------------------
def e_to_dict(ev: Any) -> Dict[str, Any]:
    """
    Convert DB Event ORM or dict to lightweight dict for modeling.
    """
    if ev is None:
        return {}
    try:
        return {
            "uid": getattr(ev, "uid", None),
            "symbol": getattr(ev, "symbol", None),
            "event_type": getattr(ev, "event_type", None),
            "text": getattr(ev, "text", None),
            "sentiment": getattr(ev, "sentiment", None),
            "relevance": getattr(ev, "relevance", None),
            "impact_hint": getattr(ev, "impact_hint", None),
            "embedding": getattr(ev, "embedding", None),
            "timestamp": getattr(ev, "timestamp", None),
            "source": getattr(ev, "source", None),
        }
    except Exception:
        try:
            return dict(ev)
        except Exception:
            return {}


def format_analysis_result(res: Dict[str, Any]) -> str:
    """Human-friendly formatting of modeling result."""
    try:
        s10 = res.get("final_score_10", 0.0)
        conf = res.get("confidence", 0.0)
        prob = res.get("prob_up")
        expected = res.get("expected_excess")
        pos = res.get("position", {}).get("size", 0.0)
        lines = [
            f"{res.get('symbol')}: Score {s10:.1f}/10 (confidence {conf:.2f})",
            f"Recommended position: {pos:.2%}",
        ]
        if prob is not None:
            lines.append(f"Model probability of positive move: {prob:.2%}")
        if expected is not None:
            lines.append(f"Model expected excess return: {expected:.2%}")
        if "mc_report" in res and res["mc_report"]:
            mc = res["mc_report"]
            median = mc.get("median", None)
            p20 = mc.get("p_ge_20pct", None) or mc.get("p_gt_20pct", None)
            if median is not None:
                lines.append(f"MC median: {median:.2%}")
            if p20 is not None:
                lines.append(f"P(>20%): {p20:.1%}")
        explain = res.get("explain", {}) or {}
        comps = explain.get("components", {}) or {}
        lines.append("Top components:")
        for k, v in comps.items():
            try:
                lines.append(f" - {k}: {v:.3f}")
            except Exception:
                lines.append(f" - {k}: {v}")
        return "\n".join(lines)
    except Exception:
        logger.exception("Failed to format analysis result")
        return str(res)


def format_mc_report(mc: Dict[str, Any]) -> str:
    """Format Monte-Carlo report in readable text."""
    if not mc:
        return "No Monte-Carlo report available."
    parts = []
    median = mc.get("median")
    worst10 = mc.get("worst_10pct")
    best10 = mc.get("best_10pct")
    if median is not None:
        parts.append(f"Median: {median:.2%}")
    if best10 is not None:
        parts.append(f"Top10%: {best10:.2%}")
    if worst10 is not None:
        parts.append(f"Worst10%: {worst10:.2%}")
    return " | ".join(parts)


async def safe_send_message(app: Any, chat_id: Optional[int], text: str) -> None:
    """
    Send message via Telegram app if present, otherwise log.
    Text will be escaped for html to be safe.
    """
    if chat_id is None:
        logger.info("No chat_id provided for message: %s", text)
        return
    try:
        escaped = html.escape(text)
        if app and getattr(app, "bot", None):
            try:
                await app.bot.send_message(chat_id=chat_id, text=escaped)
            except Exception:
                try:
                    await app.send_message(chat_id=chat_id, text=escaped)
                except Exception:
                    logger.exception("Failed to send message via telegram bot API")
        else:
            logger.info("Message to %s: %s", chat_id, text)
    except Exception:
        logger.exception("Failed to send message to %s: %s", chat_id, text)


def get_chat_id(update: "Update") -> Optional[int]:
    """Extract chat id safely from Update object (telegram)."""
    try:
        if update and hasattr(update, "effective_chat") and update.effective_chat:
            return update.effective_chat.id
        if update and hasattr(update, "message") and update.message and update.message.chat:
            return update.message.chat.id
    except Exception:
        logger.exception("Failed to get chat id from update")
    return None


# -----------------------
# Utility: maybe_await
# -----------------------
async def maybe_await(x):
    """If x is awaitable (coroutine/future) -> await it; otherwise return value."""
    if asyncio.iscoroutine(x) or asyncio.isfuture(x):
        return await x
    try:
        if hasattr(x, "__await__"):
            return await x
    except Exception:
        pass
    return x


# -----------------------
# If used as module, provide simple entrypoints
# -----------------------
async def create_and_start_orchestrator(cfg: Config):
    orch = Orchestrator(cfg)
    await orch.start()
    return orch