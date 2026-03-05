# app/mc_risk.py
"""
Monte-Carlo & Risk module for QuasiHedge Bot.

Provides a collection of Monte-Carlo simulators and risk analytics used by the orchestrator:
- GBM Monte-Carlo with drift/vol estimation and event/news shock adjustments
- Bootstrap resampling MC from historical returns
- Correlated multi-asset MC using Cholesky on empirical correlation
- Simple regime-aware MC (use regime detection to switch drift/vol)
- Portfolio-level aggregation, VaR / CVaR computation, and MC report summarization
- Helpers: annualization, percentile extraction, shock injection from events
- Config-driven, deterministic RNG optional, robust to missing data, well-logged

Design notes:
- Input price series expected as pandas.DataFrame with datetime 'timestamp' and numeric 'adj_close'
- Events list is optional; events can influence drift via impact scores (see `events_to_drift_adjust`)
- All functions are pure/functional (no DB side-effects); persist results externally if needed
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("app.mc_risk")
logger.addHandler(logging.NullHandler())


# -----------------------
# Utilities
# -----------------------
def _safe_last_price(price_df: pd.DataFrame) -> Optional[float]:
    """Return last available adj_close or None."""
    if price_df is None or price_df.empty or "adj_close" not in price_df.columns:
        return None
    try:
        return float(price_df.sort_values("timestamp").iloc[-1]["adj_close"])
    except Exception:
        return None


def _compute_log_returns(price_df: pd.DataFrame) -> pd.Series:
    """Return pandas Series of log returns indexed by timestamp."""
    if price_df is None or price_df.empty or "adj_close" not in price_df.columns:
        return pd.Series(dtype=float)
    df = price_df.sort_values("timestamp").reset_index(drop=True)
    s = pd.Series(df["adj_close"].values, index=pd.to_datetime(df["timestamp"]))
    lr = np.log(s / s.shift(1)).dropna()
    return lr


def _annualize_mean_and_vol(logrets: Iterable[float], periods_per_year: int = 252) -> Tuple[float, float]:
    """
    Convert historical log returns to annualized drift (mu) and volatility (sigma).
    Returns (mu, sigma) where mu is annualized mean log-return and sigma is annualized std dev.
    """
    arr = np.asarray(list(logrets), dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    mu = float(np.nanmean(arr)) * periods_per_year
    sigma = float(np.nanstd(arr, ddof=1)) * math.sqrt(periods_per_year)
    return mu, sigma


def _to_price_paths_from_log_returns(S0: float, logret_paths: np.ndarray) -> np.ndarray:
    """
    Convert cumulative log returns paths to price paths.
    logret_paths shape: (n_sims, n_steps) where each entry is log-return increment.
    We compute cumulative sum along axis=1 then exponentiate * S0
    """
    # cumulative log-return
    cum = np.cumsum(logret_paths, axis=1)
    price_paths = S0 * np.exp(cum)
    return price_paths


def _pct_change_to_simple_returns(price_paths: np.ndarray) -> np.ndarray:
    """Compute simple returns per path from price paths: (S_end / S_start - 1)"""
    if price_paths.size == 0:
        return np.array([])
    start = price_paths[:, 0]
    end = price_paths[:, -1]
    return end / start - 1.0


# -----------------------
# Event -> drift/vol adjustment
# -----------------------
def events_to_drift_adjust(events: Iterable[Dict[str, Any]], base_mu: float, base_sigma: float, cfg: Optional[Any] = None) -> Tuple[float, float]:
    """
    Translate list of events into additive adjustments for drift (mu) and volatility (sigma).

    Heuristic:
      - Each event contributes proportional to (sentiment * relevance * impact_hint)
      - Positive aggregated impact -> increase drift; negative -> decrease
      - Volatility adjustment proportional to abs(impact) and recency (exponential decay)

    Args:
        events: iterable of events each with keys 'sentiment', 'relevance', 'impact_hint', 'timestamp'
        base_mu: baseline annualized mu
        base_sigma: baseline annualized sigma
        cfg: optional config (controls scaling factors and half-life)

    Returns:
        (mu_adj, sigma_adj) absolute values (i.e., new_mu, new_sigma)
    """
    if events is None:
        return base_mu, base_sigma
    # config fallbacks
    half_life_days = float(getattr(cfg, "EVENT_HALF_LIFE_DAYS", 30)) if cfg is not None else 30.0
    mu_scale = float(getattr(cfg, "EVENT_MU_SCALE", 0.5)) if cfg is not None else 0.5  # how strongly events move drift (fraction of base_mu)
    sigma_scale = float(getattr(cfg, "EVENT_SIGMA_SCALE", 0.75)) if cfg is not None else 0.75  # how strongly events move vol (fractional)

    now = datetime.utcnow()
    weighted_sum = 0.0
    weight_mass = 0.0
    vol_signal = 0.0

    for e in events:
        try:
            sent = float(e.get("sentiment", 0.0) or 0.0)
            rel = float(e.get("relevance", 0.0) or 0.0)
            hint = float(e.get("impact_hint", 1.0) or 1.0)
            ts = e.get("timestamp") or e.get("event_time")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except Exception:
                    ts = now
            age_days = max(0.0, (now - ts).days) if isinstance(ts, datetime) else 0.0
            decay = math.exp(-math.log(2) * age_days / max(1.0, half_life_days))
            w = rel * abs(hint) * decay
            weighted_sum += sent * w
            weight_mass += w
            vol_signal += abs(sent) * w
        except Exception:
            logger.exception("Bad event in events_to_drift_adjust: %s", e)
            continue

    if weight_mass == 0:
        return base_mu, base_sigma

    avg_impact = weighted_sum / weight_mass
    avg_vol_signal = vol_signal / weight_mass

    # adjust mu: base_mu + mu_scale * avg_impact * |base_mu|
    mu_adj = base_mu + mu_scale * avg_impact * (abs(base_mu) if base_mu != 0 else 0.01)

    # adjust sigma: base_sigma * (1 + sigma_scale * avg_vol_signal)
    sigma_adj = base_sigma * (1.0 + sigma_scale * avg_vol_signal)

    logger.debug("events_to_drift_adjust: base_mu=%.4f base_sigma=%.4f -> mu_adj=%.4f sigma_adj=%.4f (avg_impact=%.4f)",
                 base_mu, base_sigma, mu_adj, sigma_adj, avg_impact)

    return mu_adj, sigma_adj


# -----------------------
# GBM Monte-Carlo (single-asset)
# -----------------------
def run_gbm_mc(
    cfg: Any,
    price_df: pd.DataFrame,
    horizon_days: int,
    n_sims: int = 2000,
    steps_per_day: int = 1,
    seed: Optional[int] = None,
    events: Optional[Iterable[Dict[str, Any]]] = None,
    regime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run Geometric Brownian Motion Monte-Carlo simulation.

    Args:
        cfg: config object (can supply PERIODS_PER_YEAR etc.)
        price_df: historical price DataFrame with 'timestamp' & 'adj_close'
        horizon_days: forecast horizon in days
        n_sims: number of Monte-Carlo simulation paths
        steps_per_day: resolution (1 -> daily)
        seed: RNG seed for reproducibility
        events: optional list of events to adjust drift/vol
        regime: optional regime dict (e.g., {'volatility_regime':'high'}) to scale vol

    Returns:
        dict with keys:
            - median, mean, pctiles (dict), p_ge_20pct, p_le_-10pct, worst_10pct, best_10pct
            - price_paths (not included by default to avoid huge memory; returned if cfg.RETURN_PATHS True)
            - meta: S0, mu, sigma, horizon_days, n_sims, timestamp
    """
    start_time = time.time()
    # validate input
    S0 = _safe_last_price(price_df)
    if S0 is None:
        logger.warning("run_gbm_mc: No starting price available; aborting MC")
        return {}

    # compute historical log returns to estimate mu/sigma
    logrets = _compute_log_returns(price_df)
    mu0, sigma0 = _annualize_mean_and_vol(logrets.values, periods_per_year=int(getattr(cfg, "PERIODS_PER_YEAR", 252)))
    # allow user override in cfg
    mu = float(getattr(cfg, "MC_BASE_MU", mu0))
    sigma = float(getattr(cfg, "MC_BASE_SIGMA", sigma0))

    # adjust based on events
    if events:
        mu, sigma = events_to_drift_adjust(events, mu, sigma, cfg)

    # regime scaling
    if regime:
        # simple scaling mapping; cfg.REGIME_VOL_SCALE is a dict or mapping
        regime_vol_scale = getattr(cfg, "REGIME_VOL_SCALE", {"high": 1.5, "low": 0.7, "normal": 1.0})
        vol_regime = regime.get("volatility_regime", "normal")
        scale = float(regime_vol_scale.get(vol_regime, 1.0)) if isinstance(regime_vol_scale, dict) else float(regime_vol_scale)
        sigma = sigma * scale

    # per-step parameters
    total_steps = int(max(1, horizon_days * steps_per_day))
    dt = 1.0 / (int(getattr(cfg, "PERIODS_PER_YEAR", 252)) * (1.0 / steps_per_day))
    rng = np.random.default_rng(seed)

    # drift correction for GBM: expected log-return per step = (mu - 0.5*sigma^2)/periods_per_year
    per_step_drift = (mu / int(getattr(cfg, "PERIODS_PER_YEAR", 252))) - 0.5 * (sigma ** 2) / int(getattr(cfg, "PERIODS_PER_YEAR", 252))
    per_step_vol = sigma / math.sqrt(int(getattr(cfg, "PERIODS_PER_YEAR", 252)))

    # If steps_per_day !=1 adjust dt accordingly
    # We'll treat per_step_vol and drift as per-step already when using steps_per_day by scaling with sqrt and division
    if steps_per_day != 1:
        # adjust per step
        per_step_drift = per_step_drift / float(1.0 * steps_per_day)
        per_step_vol = per_step_vol / math.sqrt(float(steps_per_day))

    # simulate log-return increments: shape (n_sims, total_steps)
    try:
        normals = rng.standard_normal(size=(n_sims, total_steps))
        logret_increments = per_step_drift + per_step_vol * normals
        # convert to price paths
        price_paths = _to_price_paths_from_log_returns(S0, logret_increments)
    except Exception:
        logger.exception("GBM simulation failed")
        return {}

    # compute endpoint simple returns (S_T / S0 -1) for each sim
    simple_returns = (price_paths[:, -1] / price_paths[:, 0]) - 1.0

    # distribution stats
    pctiles = {p: float(np.percentile(simple_returns, p)) for p in (5, 10, 25, 50, 75, 90, 95)}
    median = float(np.median(simple_returns))
    mean = float(np.mean(simple_returns))
    p_ge_20 = float((simple_returns >= 0.20).sum() / len(simple_returns))
    p_le_m10 = float((simple_returns <= -0.10).sum() / len(simple_returns))
    best_10 = float(np.percentile(simple_returns, 90))
    worst_10 = float(np.percentile(simple_returns, 10))

    elapsed = time.time() - start_time
    logger.info("GBM MC finished: S0=%.4f mu=%.4f sigma=%.4f sims=%d steps=%d elapsed=%.2fs",
                S0, mu, sigma, n_sims, total_steps, elapsed)

    report: Dict[str, Any] = {
        "S0": S0,
        "mu": mu,
        "sigma": sigma,
        "n_sims": n_sims,
        "horizon_days": horizon_days,
        "percentiles": pctiles,
        "median": median,
        "mean": mean,
        "p_ge_20pct": p_ge_20,
        "p_le_minus10pct": p_le_m10,
        "best_10pct": best_10,
        "worst_10pct": worst_10,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if getattr(cfg, "MC_RETURN_PATHS", False):
        report["price_paths"] = price_paths  # careful: large memory
    return report


# -----------------------
# Bootstrap Monte-Carlo (sampling past returns)
# -----------------------
def run_bootstrap_mc(
    cfg: Any,
    price_df: pd.DataFrame,
    horizon_days: int,
    n_sims: int = 2000,
    seed: Optional[int] = None,
    block_size: int = 1,
) -> Dict[str, Any]:
    """
    Bootstrap Monte-Carlo: resample historical log-returns with replacement (optionally block bootstrap)
    to generate future return paths (preserves empirical distribution / autocorrelation if using blocks).

    Args:
        cfg: config
        price_df: historical price DataFrame
        horizon_days: forecast horizon in days
        n_sims: number of paths
        seed: RNG seed
        block_size: block bootstrap size in days (1 -> iid bootstrap)

    Returns:
        dict similar to run_gbm_mc
    """
    start_time = time.time()
    S0 = _safe_last_price(price_df)
    if S0 is None:
        return {}

    logrets_series = _compute_log_returns(price_df)
    if logrets_series.empty:
        return {}

    rng = np.random.default_rng(seed)
    total_steps = max(1, int(horizon_days))
    # create array of logret increments for each sim
    paths = np.zeros((n_sims, total_steps), dtype=float)
    lr = logrets_series.values
    N = len(lr)
    if N == 0:
        return {}
    for i in range(n_sims):
        if block_size <= 1:
            indices = rng.integers(0, N, size=total_steps)
            sample = lr[indices]
        else:
            # block bootstrap: sample starting indices and take blocks
            sample = np.empty(total_steps, dtype=float)
            pos = 0
            while pos < total_steps:
                start_idx = int(rng.integers(0, N))
                block = lr[start_idx : start_idx + block_size]
                if block.size == 0:
                    block = lr[start_idx : start_idx + 1]
                take = min(block.size, total_steps - pos)
                sample[pos : pos + take] = block[:take]
                pos += take
        paths[i, :] = sample

    price_paths = _to_price_paths_from_log_returns(S0, paths)
    simple_returns = (price_paths[:, -1] / price_paths[:, 0]) - 1.0
    pctiles = {p: float(np.percentile(simple_returns, p)) for p in (5, 10, 25, 50, 75, 90, 95)}
    report = {
        "method": "bootstrap",
        "S0": S0,
        "n_sims": n_sims,
        "horizon_days": horizon_days,
        "percentiles": pctiles,
        "median": float(np.median(simple_returns)),
        "mean": float(np.mean(simple_returns)),
        "elapsed_seconds": time.time() - start_time,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if getattr(cfg, "MC_RETURN_PATHS", False):
        report["price_paths"] = price_paths
    return report


# -----------------------
# Correlated multi-asset MC
# -----------------------
def run_correlated_mc(
    cfg: Any,
    price_dfs: Dict[str, pd.DataFrame],
    horizon_days: int,
    n_sims: int = 2000,
    seed: Optional[int] = None,
    steps_per_day: int = 1,
) -> Dict[str, Any]:
    """
    Simulate correlated assets jointly using multivariate normal increments calibrated
    from historical log-returns covariance/correlation.

    Args:
        price_dfs: mapping symbol -> price_df
        horizon_days: forecast horizon in days
        n_sims: number of simulated paths
        seed: random seed

    Returns:
        report dict containing per-symbol percentiles and aggregated portfolio distribution if requested
    """
    start_time = time.time()
    symbols = list(price_dfs.keys())
    if not symbols:
        return {}

    # compute aligned log-return series for all symbols by inner-join on timestamps
    series_list = []
    for s in symbols:
        lr = _compute_log_returns(price_dfs[s]).rename(s)
        series_list.append(lr)
    if not series_list:
        return {}
    df_lr = pd.concat(series_list, axis=1, join="inner").dropna()
    if df_lr.empty:
        # fallback: align by index union with forward/backfill
        df_lr = pd.concat(series_list, axis=1).dropna(how="all").fillna(0.0)

    # estimate mean vector and covariance
    mu_vec = df_lr.mean(axis=0).values  # per-period mean log-returns
    cov = df_lr.cov().values
    # annualize
    periods_per_year = int(getattr(cfg, "PERIODS_PER_YEAR", 252))
    mu_annual = mu_vec * periods_per_year
    sigma_annual = np.sqrt(np.diag(cov) * periods_per_year)

    rng = np.random.default_rng(seed)
    total_steps = max(1, int(horizon_days * steps_per_day))
    # For simplicity simulate at per-day granularity using multivariate normals with per-step cov = cov / periods_per_year
    per_step_cov = cov / periods_per_year
    try:
        # cholesky
        L = np.linalg.cholesky(per_step_cov)
    except np.linalg.LinAlgError:
        # fallback to near PSD adjustment (add small diag)
        eps = 1e-8
        try:
            L = np.linalg.cholesky(per_step_cov + np.eye(per_step_cov.shape[0]) * eps)
        except Exception:
            logger.exception("Cholesky failed for covariance matrix; aborting correlated MC")
            return {}

    # generate standard normals and transform
    sims_price_paths = {s: None for s in symbols}
    # Simulate increments for each step: Z @ L.T -> correlated normals
    all_paths = np.zeros((n_sims, total_steps, len(symbols)), dtype=float)
    for step in range(total_steps):
        Z = rng.standard_normal(size=(n_sims, len(symbols)))  # shape (n_sims, n_assets)
        correlated = Z @ L.T  # shape (n_sims, n_assets)
        # per-step log-return = mu_per_step + correlated*1
        per_step_mu = mu_vec / periods_per_year
        all_paths[:, step, :] = per_step_mu + correlated

    # convert to price paths per asset
    reports = {}
    for idx, s in enumerate(symbols):
        S0 = _safe_last_price(price_dfs[s])
        if S0 is None:
            reports[s] = {}
            continue
        # extract increments for asset idx -> shape (n_sims, total_steps)
        increments = all_paths[:, :, idx]
        price_paths = _to_price_paths_from_log_returns(S0, increments)
        sim_returns = (price_paths[:, -1] / price_paths[:, 0]) - 1.0
        pctiles = {p: float(np.percentile(sim_returns, p)) for p in (5, 10, 25, 50, 75, 90, 95)}
        reports[s] = {
            "S0": S0,
            "median": float(np.median(sim_returns)),
            "mean": float(np.mean(sim_returns)),
            "percentiles": pctiles,
        }
        if getattr(cfg, "MC_RETURN_PATHS", False):
            reports[s]["price_paths"] = price_paths

    elapsed = time.time() - start_time
    return {"method": "correlated", "symbols": symbols, "per_asset": reports, "elapsed_seconds": elapsed, "timestamp": datetime.utcnow().isoformat()}


# -----------------------
# Value-at-Risk & CVaR
# -----------------------
def compute_var_cvar(returns: Iterable[float], alpha: float = 0.95) -> Dict[str, float]:
    """
    Compute VaR and CVaR (expected shortfall) for a series of simple returns.

    Args:
        returns: iterable of realized or simulated returns (simple returns)
        alpha: confidence level (e.g., 0.95)

    Returns:
        dict { 'var': value (positive), 'cvar': value }
    """
    arr = np.asarray(list(returns), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"var": 0.0, "cvar": 0.0}
    # VaR at alpha is negative quantile for losses; we will report positive numbers as absolute loss
    q = np.quantile(arr, 1.0 - alpha)
    var = -float(q) if q < 0 else 0.0
    # CVaR: mean loss beyond VaR
    tail = arr[arr <= q]
    cvar = -float(np.mean(tail)) if tail.size > 0 else var
    return {"var": var, "cvar": cvar}


# -----------------------
# Summarize MC report & helper to run ensemble MC
# -----------------------
def summarize_simulation_simple(sim_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and return a concise summary from a run_gbm_mc or run_bootstrap_mc report.
    """
    if not sim_report:
        return {}
    summary = {
        "median": sim_report.get("median"),
        "mean": sim_report.get("mean"),
        "pct_5": sim_report.get("percentiles", {}).get(5),
        "pct_95": sim_report.get("percentiles", {}).get(95),
        "p_ge_20pct": sim_report.get("p_ge_20pct"),
    }
    return summary


def run_ensemble_mc(
    cfg: Any,
    price_df: pd.DataFrame,
    horizon_days: int,
    n_sims: int = 2000,
    methods: Optional[Iterable[str]] = None,
    seed: Optional[int] = None,
    events: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run ensemble of MC methods (GBM, bootstrap, maybe regime) and aggregate results.

    Args:
        methods: list of methods e.g., ['gbm','bootstrap']

    Returns:
        dict combining individual reports and an ensemble summary
    """
    if methods is None:
        methods = ["gbm", "bootstrap"]
    reports = {}
    rng_seed = seed or int(time.time() % 2 ** 31)
    for idx, m in enumerate(methods):
        s = rng_seed + idx * 13
        if m == "gbm":
            r = run_gbm_mc(cfg, price_df, horizon_days, n_sims=n_sims, seed=s, events=events)
        elif m == "bootstrap":
            r = run_bootstrap_mc(cfg, price_df, horizon_days, n_sims=n_sims, seed=s)
        else:
            logger.warning("Unknown MC method '%s' requested; skipping", m)
            r = {}
        reports[m] = r

    # simple ensemble: average percentiles across methods (where present)
    percentiles = {}
    for p in (5, 10, 25, 50, 75, 90, 95):
        vals = []
        for r in reports.values():
            if r and isinstance(r.get("percentiles", {}), dict):
                v = r["percentiles"].get(p)
                if v is not None:
                    vals.append(v)
        if vals:
            percentiles[p] = float(np.mean(vals))
    # compute ensemble p_ge_20 as mean
    p_ge_20s = [r.get("p_ge_20pct") for r in reports.values() if r.get("p_ge_20pct") is not None]
    ensemble_p_ge_20 = float(np.mean(p_ge_20s)) if p_ge_20s else None

    ensemble = {
        "methods": list(reports.keys()),
        "percentiles": percentiles,
        "p_ge_20pct": ensemble_p_ge_20,
        "per_method": reports,
        "timestamp": datetime.utcnow().isoformat(),
    }
    return ensemble


# -----------------------
# Portfolio MC (weights + correlated sims) and risk metrics
# -----------------------
def run_portfolio_mc(
    cfg: Any,
    price_dfs: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    horizon_days: int,
    n_sims: int = 2000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Simulate portfolio returns by running correlated MC and computing portfolio distribution.

    Args:
        price_dfs: mapping symbol -> price_df
        weights: mapping symbol -> portfolio weight (sums to <=1)
        horizon_days: horizon in days

    Returns:
        dict with portfolio percentiles, VaR/CVaR, per-asset reports
    """
    corr_report = run_correlated_mc(cfg, price_dfs, horizon_days, n_sims=n_sims, seed=seed)
    if not corr_report or corr_report.get("method") != "correlated":
        return {}
    symbols = corr_report.get("symbols", [])
    # gather price paths per symbol if present, otherwise recompute with MC_RETURN_PATHS True
    # For memory reasons, correlated MC may not return paths by default; recompute with paths if needed
    total_paths = None
    need_paths = True
    per_asset = corr_report.get("per_asset", {})
    # If per_asset entries have price_paths, use them
    has_paths = all(per_asset.get(s, {}).get("price_paths") is not None for s in symbols)
    if has_paths:
        # build array shape (n_sims, steps, n_assets)
        first_sym = symbols[0]
        price_paths_stack = []
        for s in symbols:
            price_paths_stack.append(per_asset[s]["price_paths"])
        # verify shapes
        try:
            arr = np.stack(price_paths_stack, axis=2)  # shape (n_sims, steps, n_assets)
            total_paths = arr
        except Exception:
            total_paths = None

    if total_paths is None:
        # fallback: run correlated MC again with MC_RETURN_PATHS True
        old_flag = getattr(cfg, "MC_RETURN_PATHS", False)
        setattr(cfg, "MC_RETURN_PATHS", True)
        corr_report2 = run_correlated_mc(cfg, price_dfs, horizon_days, n_sims=n_sims, seed=seed)
        setattr(cfg, "MC_RETURN_PATHS", old_flag)
        per_asset2 = corr_report2.get("per_asset", {})
        price_paths_stack = []
        for s in symbols:
            price_paths_stack.append(per_asset2.get(s, {}).get("price_paths"))
        try:
            arr = np.stack(price_paths_stack, axis=2)
            total_paths = arr
        except Exception:
            logger.exception("Failed to assemble portfolio price paths")
            return {}

    # compute portfolio value paths given weights (weights must align with symbol order)
    w = np.array([weights.get(s, 0.0) for s in symbols], dtype=float)
    # initial portfolio value normalized to 1.0
    # Convert price paths to returns per asset relative to start, then weighted sum
    # price_paths shape (n_sims, steps, n_assets)
    rel_returns = total_paths / total_paths[:, 0:1, :] - 1.0  # relative to t0
    # take final returns
    final_rel = rel_returns[:, -1, :]  # shape (n_sims, n_assets)
    # portfolio returns per sim = dot(final_rel, w)
    port_returns = final_rel.dot(w)
    pctiles = {p: float(np.percentile(port_returns, p)) for p in (5, 10, 25, 50, 75, 90, 95)}
    var_cvar = compute_var_cvar(port_returns, alpha=float(getattr(cfg, "VAR_ALPHA", 0.95)))
    report = {
        "method": "portfolio_correlated",
        "symbols": symbols,
        "weights": weights,
        "port_returns": {"median": float(np.median(port_returns)), "mean": float(np.mean(port_returns)), "percentiles": pctiles},
        "var": var_cvar.get("var"),
        "cvar": var_cvar.get("cvar"),
        "timestamp": datetime.utcnow().isoformat(),
    }
    return report


# -----------------------
# Convenience wrapper used by core to run a standard MC and return compact report
# -----------------------
def mc_pipeline_for_core(
    cfg: Any,
    price_df: pd.DataFrame,
    events: Optional[Iterable[Dict[str, Any]]] = None,
    horizon_days: int = 365,
    n_sims: int = 2000,
    method: str = "ensemble",
    seed: Optional[int] = None,
    regime_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    High-level pipeline invoked by core._analyze_and_report:
    - detect regime from price_df
    - choose simulation method(s)
    - apply events as drift/vol shocks
    - return compact MC report dictionary

    Returns:
        report dict with ensemble or single method results
    """
    # regime detection can be done by modeling.detect_market_regime if available; we accept regime_override
    regime = regime_override or getattr(cfg, "REGIME_DETECT_OVERRIDE", None)
    # choose methods
    if method == "gbm":
        rep = run_gbm_mc(cfg, price_df, horizon_days, n_sims=n_sims, seed=seed, events=events, regime=regime)
        return rep
    elif method == "bootstrap":
        rep = run_bootstrap_mc(cfg, price_df, horizon_days, n_sims=n_sims, seed=seed)
        return rep
    else:
        # ensemble default
        rep = run_ensemble_mc(cfg, price_df, horizon_days, n_sims=n_sims, seed=seed, methods=getattr(cfg, "MC_ENSEMBLE_METHODS", ["gbm", "bootstrap"]), events=events)
        return rep