# app/modeling.py
"""
Modeling and scoring module for quasihedge-bot.

Provides:
- Fundamental / market / institutional factor computation
- Event aggregation -> event_score
- Regime detection
- Feature engineering for event->reaction models
- Predict / train models (LightGBM / sklearn fallbacks)
- Explainability (SHAP if available)
- Risk metrics (Sharpe, Sortino, Max Drawdown, Beta)
- Cross-sectional ranking and position sizing (Kelly-like + vol-scaling)

Design:
- Config-driven defaults (via cfg object)
- Robust to missing optional dependencies (lightgbm, shap, sklearn)
- Works with pandas DataFrame price history and dict-style fundamentals/events
"""
from __future__ import annotations

import logging
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Optional ML libraries
try:
    import lightgbm as lgb  # type: ignore
    _HAS_LGB = True
except Exception:
    lgb = None  # type: ignore
    _HAS_LGB = False

try:
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import roc_auc_score, mean_squared_error  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    RandomForestRegressor = None  # type: ignore
    LogisticRegression = None  # type: ignore
    train_test_split = None  # type: ignore
    roc_auc_score = None  # type: ignore
    mean_squared_error = None  # type: ignore
    _HAS_SKLEARN = False

# Explainability
try:
    import shap  # type: ignore

    _HAS_SHAP = True
except Exception:
    shap = None  # type: ignore
    _HAS_SHAP = False

# Local modules
import app.db as db
import app.nlp_events as nlp

logger = logging.getLogger("app.modeling")
logger.addHandler(logging.NullHandler())

# ---------- helpers and small utilities ----------


def _safe_mean(xs: Sequence[float]) -> Optional[float]:
    """Return mean or None if no valid items."""
    arr = [float(x) for x in xs if x is not None and not math.isnan(x)]
    if not arr:
        return None
    return float(sum(arr) / len(arr))


def _annualize_return_from_log_returns(log_returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Convert mean log-return per period to annualized return."""
    if not log_returns:
        return 0.0
    mu = float(np.nanmean(log_returns))
    return mu * periods_per_year


def _annualize_vol_from_log_returns(log_returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Annualize volatility (std) from period log returns."""
    if not log_returns:
        return 0.0
    sigma = float(np.nanstd(log_returns))
    return sigma * np.sqrt(periods_per_year)


# ---------- Financial / Risk Metrics ----------


def compute_log_returns_from_prices(df: pd.DataFrame, price_col: str = "adj_close") -> pd.Series:
    """
    Compute log returns series from price DataFrame.

    Args:
        df: DataFrame with datetime index or 'timestamp' column and price_col
        price_col: column name for price

    Returns:
        pd.Series of log returns (aligned with df index starting at 1)
    """
    if df is None or df.empty or price_col not in df.columns:
        return pd.Series(dtype=float)
    prices = pd.Series(df[price_col].values, index=pd.to_datetime(df["timestamp"]))
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def sharpe_ratio(returns: Sequence[float], risk_free: float = 0.0, annualize: bool = True, periods_per_year: int = 252) -> Optional[float]:
    """
    Compute Sharpe ratio (annualized if annualize=True).

    Args:
        returns: sequence of period returns (not log returns)
        risk_free: per-period risk-free rate (if annualize True, risk_free should be annual)
        annualize: whether to annualize
        periods_per_year: periods per year for annualization

    Returns:
        Sharpe ratio or None if insufficient data
    """
    arr = np.array([r for r in returns if r is not None and not math.isnan(r)])
    if arr.size < 2:
        return None
    if annualize:
        # convert to simple returns if given log returns? assume simple returns
        mean = arr.mean() * periods_per_year
        vol = arr.std(ddof=1) * math.sqrt(periods_per_year)
    else:
        mean = arr.mean()
        vol = arr.std(ddof=1)
    if vol == 0:
        return None
    rf = risk_free if not annualize else risk_free
    return float((mean - rf) / vol)


def sortino_ratio(returns: Sequence[float], required_return: float = 0.0, periods_per_year: int = 252) -> Optional[float]:
    """
    Compute Sortino ratio.

    Args:
        returns: sequence of period returns
        required_return: minimal acceptable return (annual if these are annual)
    """
    arr = np.array([r for r in returns if r is not None and not math.isnan(r)])
    if arr.size < 2:
        return None
    # downside deviation
    downside = arr[arr < required_return]
    if downside.size == 0:
        # no downside => very large
        return float("inf")
    dd = np.sqrt(np.mean((downside - required_return) ** 2))
    mean = arr.mean() * 252
    if dd == 0:
        return None
    return float((mean - required_return) / (dd * math.sqrt(252)))


def max_drawdown(prices: Sequence[float]) -> Optional[float]:
    """
    Compute maximum drawdown for a price series.

    Returns fraction (e.g., 0.25 means 25% drawdown).
    """
    if not prices:
        return None
    arr = np.asarray([p for p in prices if p is not None and not math.isnan(p)], dtype=float)
    if arr.size == 0:
        return None
    peak = arr[0]
    max_dd = 0.0
    for v in arr:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak != 0 else 0.0
        max_dd = max(max_dd, dd)
    return float(max_dd)


def beta_from_returns(asset_returns: Sequence[float], benchmark_returns: Sequence[float]) -> Optional[float]:
    """
    Compute beta of asset vs benchmark given aligned period returns.
    """
    a = np.array(asset_returns)
    b = np.array(benchmark_returns)
    # align lengths
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return None
    cov = np.cov(a, b, ddof=1)
    if cov.shape != (2, 2):
        return None
    var_b = cov[1, 1]
    if var_b == 0:
        return None
    beta = cov[0, 1] / var_b
    return float(beta)


# ---------- Regime detection ----------


def detect_market_regime(price_df: pd.DataFrame, cfg: Any) -> Dict[str, Any]:
    """
    Detect market regime based on volatility and trend rules.

    Returns dict: {'volatility_regime': 'low'|'normal'|'high', 'trend_regime': 'up'|'down'|'sideways', 'combined': '...'}
    """
    if price_df is None or price_df.empty:
        return {"volatility_regime": "unknown", "trend_regime": "unknown", "combined": "unknown"}

    # default lookbacks from cfg
    vol_lookback = int(getattr(cfg, "REGIME_VOL_LOOKBACK_DAYS", 252))
    trend_fast = int(getattr(cfg, "REGIME_TREND_FAST", 50))
    trend_slow = int(getattr(cfg, "REGIME_TREND_SLOW", 200))

    df = price_df.copy().sort_values("timestamp")
    # compute daily log returns
    df["logret"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    df = df.dropna(subset=["logret"])
    if df.empty:
        return {"volatility_regime": "unknown", "trend_regime": "unknown", "combined": "unknown"}

    # volatility regime
    recent_vol = df["logret"].rolling(window=min(len(df), vol_lookback)).std().iloc[-1]
    hist_vol = df["logret"].std()
    if math.isnan(recent_vol) or math.isnan(hist_vol) or hist_vol == 0:
        vol_regime = "unknown"
    else:
        if recent_vol > hist_vol * 1.5:
            vol_regime = "high"
        elif recent_vol < hist_vol * 0.7:
            vol_regime = "low"
        else:
            vol_regime = "normal"

    # trend regime using SMA slopes
    df["sma_fast"] = df["adj_close"].rolling(window=min(len(df), trend_fast)).mean()
    df["sma_slow"] = df["adj_close"].rolling(window=min(len(df), trend_slow)).mean()
    # take last valid
    try:
        sma_fast = df["sma_fast"].dropna().iloc[-1]
        sma_slow = df["sma_slow"].dropna().iloc[-1]
        if sma_fast > sma_slow:
            trend_regime = "up"
        elif sma_fast < sma_slow:
            trend_regime = "down"
        else:
            trend_regime = "sideways"
    except Exception:
        trend_regime = "unknown"

    combined = f"{trend_regime}_{vol_regime}"
    return {"volatility_regime": vol_regime, "trend_regime": trend_regime, "combined": combined}


# ---------- Factor computations ----------


def compute_fundamental_score(fundamentals: Dict[str, Any], cfg: Any) -> Tuple[float, Dict[str, float]]:
    """
    Compute a normalized fundamental score in [0,1] based on provided fundamental metrics.

    Args:
        fundamentals: dict with keys like revenue_yoy, gross_margin, fcf, debt_to_equity, roe, pe, ps, pb
        cfg: config with weighting params optional

    Returns:
        (score, detail_factors)
    """
    # defensive defaults
    if fundamentals is None:
        fundamentals = {}
    # extract metrics with fallbacks
    revenue_yoy = float(fundamentals.get("revenue_yoy", 0.0))
    gross_margin = float(fundamentals.get("gross_margin", 0.0))
    fcf = float(fundamentals.get("free_cash_flow", 0.0))
    debt_eq = float(fundamentals.get("debt_to_equity", 0.0))
    roe = float(fundamentals.get("roe", 0.0))
    pe = fundamentals.get("pe")
    ps = fundamentals.get("ps")
    # score each component to 0..1
    def score_range(val, good=1.0, bad=0.0, clamp_min=-10, clamp_max=10):
        # linear mapping between bad and good; caller scales accordingly
        try:
            v = float(val)
        except Exception:
            return 0.5
        # simple logistic-ish mapping
        if v <= bad:
            return 0.0
        # normalize using arctan
        scaled = (math.atan(v) / (math.pi / 2))
        return float(max(0.0, min(1.0, scaled)))

    # For better interpretability, use explicit transformations:
    rev_score = max(0.0, min(1.0, (revenue_yoy / 50.0)))  # 50% yoy -> 1.0
    margin_score = max(0.0, min(1.0, gross_margin / 50.0))  # 50% margin -> 1
    fcf_score = 1.0 if fcf > 0 else 0.0
    debt_score = max(0.0, min(1.0, 1.0 - debt_eq / 2.0))  # debt/equity 0 -> 1, 2 -> 0
    roe_score = max(0.0, min(1.0, roe / 20.0))  # 20% ROE -> 1
    pe_score = 1.0 if (pe is None) else max(0.0, min(1.0, 1.0 - (float(pe) / 100.0)))  # 100 P/E -> 0
    # combine with configurable weights
    w_rev = float(getattr(cfg, "FUND_W_REV", 0.25))
    w_margin = float(getattr(cfg, "FUND_W_MARGIN", 0.2))
    w_fcf = float(getattr(cfg, "FUND_W_FCF", 0.2))
    w_debt = float(getattr(cfg, "FUND_W_DEBT", 0.15))
    w_roe = float(getattr(cfg, "FUND_W_ROE", 0.1))
    w_pe = float(getattr(cfg, "FUND_W_PE", 0.1))
    # normalize weights
    total = w_rev + w_margin + w_fcf + w_debt + w_roe + w_pe
    if total <= 0:
        total = 1.0
    w_rev /= total
    w_margin /= total
    w_fcf /= total
    w_debt /= total
    w_roe /= total
    w_pe /= total

    score = (
        rev_score * w_rev
        + margin_score * w_margin
        + fcf_score * w_fcf
        + debt_score * w_debt
        + roe_score * w_roe
        + pe_score * w_pe
    )
    details = {
        "revenue_score": rev_score,
        "margin_score": margin_score,
        "fcf_score": fcf_score,
        "debt_score": debt_score,
        "roe_score": roe_score,
        "pe_score": pe_score,
    }
    return float(max(0.0, min(1.0, score))), details


def compute_market_features(price_df: pd.DataFrame, cfg: Any) -> Dict[str, Any]:
    """
    Compute market features: momentum (1/3/6/12M), volatility (30/90), avg daily volume, liquidity proxy,
    vwap deviation, sma slopes, hurst exponent, z-scores.

    Returns dict of features (floats).
    """
    features: Dict[str, Any] = {}
    if price_df is None or price_df.empty or "adj_close" not in price_df.columns:
        logger.debug("compute_market_features: empty price_df")
        return features

    df = price_df.copy().sort_values("timestamp").reset_index(drop=True)
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df = df.dropna(subset=["adj_close"])
    if df.empty:
        return features

    # compute returns and volumes
    df["ret_1"] = df["adj_close"].pct_change()
    df["logret"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    # momentums
    def pct_change_n(n_days):
        if len(df) < n_days:
            return float(df["adj_close"].iloc[-1] / df["adj_close"].iloc[0] - 1.0)
        return float(df["adj_close"].iloc[-1] / df["adj_close"].iloc[-n_days] - 1.0)

    # approximate days => periods mapping (assuming daily data)
    N1 = min(len(df), 21)  # ~1 month
    N3 = min(len(df), 63)  # ~3 months
    N6 = min(len(df), 126)
    N12 = min(len(df), 252)
    features["momentum_1m"] = pct_change_n(N1)
    features["momentum_3m"] = pct_change_n(N3)
    features["momentum_6m"] = pct_change_n(N6)
    features["momentum_12m"] = pct_change_n(N12)
    # volatility (rolling std of log returns)
    features["vol_30d"] = float(df["logret"].rolling(window=min(30, len(df))).std().iloc[-1] * math.sqrt(252)) if len(df) > 1 else 0.0
    features["vol_90d"] = float(df["logret"].rolling(window=min(90, len(df))).std().iloc[-1] * math.sqrt(252)) if len(df) > 1 else 0.0
    # avg daily volume
    if "volume" in df.columns:
        features["avg_vol_30d"] = float(df["volume"].rolling(window=min(30, len(df))).mean().iloc[-1]) if len(df) > 0 else 0.0
        # liquidity proxy: avg_vol * price
        features["liquidity_proxy"] = float(features["avg_vol_30d"] * df["adj_close"].iloc[-1])
    else:
        features["avg_vol_30d"] = 0.0
        features["liquidity_proxy"] = 0.0
    # SMA slopes
    df["sma20"] = df["adj_close"].rolling(window=min(20, len(df))).mean()
    df["sma50"] = df["adj_close"].rolling(window=min(50, len(df))).mean()
    df["sma200"] = df["adj_close"].rolling(window=min(200, len(df))).mean()
    try:
        features["sma20_slope"] = float((df["sma20"].iloc[-1] - df["sma20"].iloc[-5]) / df["sma20"].iloc[-5]) if len(df) > 5 else 0.0
    except Exception:
        features["sma20_slope"] = 0.0
    try:
        features["sma50_over_200"] = float(df["sma50"].iloc[-1] / df["sma200"].iloc[-1]) if df["sma200"].iloc[-1] not in (0, None, np.nan) else 1.0
    except Exception:
        features["sma50_over_200"] = 1.0

    # VWAP deviation: (last price - vwap_20) / vwap_20
    if "volume" in df.columns:
        try:
            vwap = (df["adj_close"] * df["volume"]).rolling(window=min(20, len(df))).sum() / df["volume"].rolling(window=min(20, len(df))).sum()
            features["vwap_dev_20"] = float((df["adj_close"].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]) if vwap.iloc[-1] not in (0, None, np.nan) else 0.0
        except Exception:
            features["vwap_dev_20"] = 0.0
    else:
        features["vwap_dev_20"] = 0.0

    # Hurst exponent (simple estimate)
    def hurst_exponent(ts):
        try:
            ts = np.array(ts.dropna(), dtype=float)
            N = len(ts)
            if N < 20:
                return 0.5
            lags = range(2, min(100, N // 2))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            m = np.polyfit(np.log(lags), np.log(tau), 1)
            return float(2.0 * m[0])
        except Exception:
            return 0.5

    features["hurst"] = hurst_exponent(df["adj_close"])

    # z-score of last return relative to 1y
    try:
        recent = df["logret"].dropna()
        if len(recent) >= 10:
            z = (recent.iloc[-1] - recent.mean()) / recent.std(ddof=1) if recent.std(ddof=1) != 0 else 0.0
            features["zscore_last"] = float(z)
        else:
            features["zscore_last"] = 0.0
    except Exception:
        features["zscore_last"] = 0.0

    return features


# ---------- Event aggregation & scoring ----------


def aggregate_event_score(events: Sequence[Dict[str, Any]], cfg: Any) -> Tuple[float, Dict[str, Any]]:
    """
    Aggregate multiple events (from db.find_analogs or saved events) into a single event_score [0,1] and details.

    Each event dict expected to have keys: 'sentiment', 'relevance', 'impact_hint', 'timestamp'.

    Aggregation logic:
    - For each event compute weighted_impact = sentiment * relevance * exp_decay(age_days) * impact_hint_weight
    - Sum positive and negative separately, combine to normalized score

    Returns:
        (score, details) where score in [0,1], details include raw sums and counts.
    """
    if not events:
        return 0.0, {"n": 0}
    now = datetime.utcnow()
    half_life = float(getattr(cfg, "EVENT_HALF_LIFE_DAYS", 30))
    pos_sum = 0.0
    neg_sum = 0.0
    weights = []
    for e in events:
        try:
            sent = float(e.get("sentiment", 0.0) or 0.0)
            rel = float(e.get("relevance", 0.0) or 0.0)
            hint = float(e.get("impact_hint", 0.0) or 1.0)
            ts = e.get("timestamp") or e.get("event_time") or now
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except Exception:
                    ts = now
            age_days = max(0.0, (now - ts).days if isinstance(ts, datetime) else 0.0)
            decay = math.exp(-math.log(2) * age_days / half_life) if half_life > 0 else 1.0
            w = rel * decay * (abs(hint) if hint is not None else 1.0)
            weights.append(w)
            impact = sent * w
            if impact >= 0:
                pos_sum += impact
            else:
                neg_sum += abs(impact)
        except Exception:
            logger.exception("Failed to aggregate event: %s", getattr(e, "id", "<unknown>"))
            continue
    # normalize to 0..1 by ratio pos / (pos+neg+eps)
    eps = 1e-9
    total = pos_sum + neg_sum + eps
    raw_score = pos_sum / total
    # map raw_score [0,1] to center 0.5 neutral, produce final event score 0..1
    # we can also scale by average weight to capture magnitude
    avg_w = float(sum(weights) / len(weights)) if weights else 0.0
    # amplify score if strong avg weight
    amplify = min(2.0, 1.0 + avg_w)
    final = max(0.0, min(1.0, (raw_score - 0.5) * amplify + 0.5))
    details = {"n": len(events), "pos_sum": pos_sum, "neg_sum": neg_sum, "avg_weight": avg_w, "raw_score": raw_score}
    return float(final), details


# ---------- Feature engineering for ML ----------


def build_feature_vector(
    *,
    cfg: Any,
    symbol: str,
    price_df: pd.DataFrame,
    fundamentals: Optional[Dict[str, Any]] = None,
    events: Optional[Sequence[Dict[str, Any]]] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Build a dictionary of features for the given asset by combining
    market, fundamental and aggregated event features.

    Returns:
        features_dict (flat mapping), feature_names_list
    """
    features: Dict[str, float] = {}
    # market features
    market_feats = compute_market_features(price_df, cfg)
    features.update({f"m_{k}": float(v) for k, v in market_feats.items()})

    # fundamental score and breakdown
    fund_score, fund_details = compute_fundamental_score(fundamentals or {}, cfg)
    features["fund_score"] = fund_score
    for k, v in fund_details.items():
        features[f"fund_{k}"] = float(v)

    # event aggregation
    evt_score, evt_details = aggregate_event_score(events or [], cfg)
    features["event_score"] = evt_score
    features.update({f"evt_{k}": float(v) if v is not None else 0.0 for k, v in evt_details.items()})

    # risk metrics from price history
    logrets = compute_log_returns_from_prices(price_df)
    features["annualized_vol"] = _annualize_vol_from_log_returns(logrets) if not logrets.empty else 0.0

    # beta relative to benchmark if provided
    if benchmark_df is not None and not benchmark_df.empty:
        bench_logrets = compute_log_returns_from_prices(benchmark_df)
        # align lengths by trimming to same end
        a = logrets.values if not logrets.empty else np.array([])
        b = bench_logrets.values if not bench_logrets.empty else np.array([])
        # align last N
        if a.size and b.size:
            N = min(len(a), len(b))
            beta_val = beta_from_returns(a[-N:], b[-N:]) if N > 1 else None
            features["beta"] = float(beta_val) if beta_val is not None else 0.0
        else:
            features["beta"] = 0.0
    else:
        features["beta"] = 0.0

    # additional heuristics
    features["liquidity_proxy"] = float(market_feats.get("liquidity_proxy", 0.0))
    features["momentum_6m"] = float(market_feats.get("momentum_6m", 0.0))
    # return feature names
    return features, list(features.keys())


# ---------- ML model predict / train ----------


@dataclass
class EventReactionModel:
    """Wrapper for loaded model and metadata"""
    model: Any
    feature_names: List[str]
    model_type: str = "lightgbm"
    version: Optional[str] = None

    def predict_probs(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of positive significant move (binary)"""
        if self.model_type == "lightgbm" and _HAS_LGB:
            preds = self.model.predict(X[self.feature_names], num_iteration=self.model.best_iteration)
            # if model returns continuous expected return we map via logistic or threshold externally
            return np.asarray(preds)
        elif _HAS_SKLEARN:
            try:
                return np.asarray(self.model.predict_proba(X[self.feature_names])[:, 1])
            except Exception:
                return np.asarray(self.model.predict(X[self.feature_names]))
        else:
            raise RuntimeError("No ML backend available")

    def predict_expected_return(self, X: pd.DataFrame) -> np.ndarray:
        """Return expected excess return (regression mode) if model supports"""
        if self.model_type == "lightgbm" and _HAS_LGB:
            return np.asarray(self.model.predict(X[self.feature_names], num_iteration=self.model.best_iteration))
        elif _HAS_SKLEARN:
            try:
                return np.asarray(self.model.predict(X[self.feature_names]))
            except Exception:
                raise
        else:
            raise RuntimeError("No ML backend available")


def load_event_reaction_model(path: str) -> Optional[EventReactionModel]:
    """
    Load model from file system. Supports LightGBM Booster (.txt/.bin via lgb.Booster or sklearn joblib).

    Returns:
        EventReactionModel or None
    """
    if not path or not os.path.exists(path):
        logger.warning("Model path does not exist: %s", path)
        return None
    try:
        if _HAS_LGB and path.endswith((".txt", ".bin", ".model")):
            model = lgb.Booster(model_file=path)
            # feature names from model
            fns = list(model.feature_name())
            return EventReactionModel(model=model, feature_names=fns, model_type="lightgbm", version=None)
        elif _HAS_SKLEARN:
            import joblib  # type: ignore

            model = joblib.load(path)
            # we require feature_names attribute stored separately
            if hasattr(model, "feature_names_in_"):
                fns = list(model.feature_names_in_)
            else:
                fns = getattr(model, "feature_names", [])
            return EventReactionModel(model=model, feature_names=fns, model_type="sklearn", version=None)
        else:
            logger.error("No supported model runtime available to load %s", path)
            return None
    except Exception:
        logger.exception("Failed to load model from %s", path)
        return None


def train_event_reaction_model(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    cfg: Any,
    save_path: Optional[str] = None,
    model_type: str = "lightgbm",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Optional[EventReactionModel], Dict[str, Any]]:
    """
    Train a model to predict event→reaction (binary or regression as y).

    Args:
        X: feature DataFrame
        y: target (binary or continuous)
        cfg: config
        save_path: where to save trained model
        model_type: 'lightgbm' or 'sklearn_rf'
    Returns:
        (EventReactionModel, metrics)
    """
    metrics: Dict[str, Any] = {}
    if X is None or X.empty:
        logger.error("train_event_reaction_model: empty X")
        return None, metrics
    try:
        if _HAS_SKLEARN:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            # no sklearn: use full data
            X_train, y_train = X, y
            X_test, y_test = X, y

        if model_type == "lightgbm" and _HAS_LGB:
            lgb_train = lgb.Dataset(X_train.values, label=y_train)
            lgb_eval = lgb.Dataset(X_test.values, label=y_test, reference=lgb_train)
            params = {
                "objective": "regression" if np.issubdtype(type(y[0]), np.floating) else "binary",
                "metric": "rmse" if np.issubdtype(type(y[0]), np.floating) else "binary_logloss",
                "verbosity": -1,
            }
            num_round = int(getattr(cfg, "LGB_NUM_ROUNDS", 100))
            model = lgb.train(params, lgb_train, num_boost_round=num_round, valid_sets=[lgb_eval], verbose_eval=False)
            er_model = EventReactionModel(model=model, feature_names=list(X.columns), model_type="lightgbm")
            # save model if requested
            if save_path:
                model.save_model(save_path)
            # eval metrics
            preds = model.predict(X_test.values)
            if np.issubdtype(preds.dtype, np.floating):
                metrics["rmse"] = float(mean_squared_error(y_test, preds) ** 0.5) if mean_squared_error else None
            else:
                metrics["auc"] = float(roc_auc_score(y_test, preds)) if roc_auc_score else None
            return er_model, metrics
        elif _HAS_SKLEARN:
            # use RandomForestRegressor as fallback (works for regression and classification via predict_proba)
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            model.fit(X_train, y_train)
            er_model = EventReactionModel(model=model, feature_names=list(X.columns), model_type="sklearn")
            if save_path:
                import joblib  # type: ignore

                joblib.dump(model, save_path)
            preds = model.predict(X_test)
            metrics["rmse"] = float(mean_squared_error(y_test, preds)) if mean_squared_error else None
            return er_model, metrics
        else:
            logger.error("No ML backend available for training")
            return None, metrics
    except Exception:
        logger.exception("train_event_reaction_model failed")
        return None, metrics


# ---------- Explainability ----------


def explain_model_predictions(er_model: EventReactionModel, X: pd.DataFrame, max_display: int = 10) -> Dict[str, Any]:
    """
    Explain predictions using SHAP if available; otherwise use feature importances.

    Returns a dict with explanation summary.
    """
    try:
        if _HAS_SHAP and er_model.model is not None:
            try:
                explainer = shap.Explainer(er_model.model, X[er_model.feature_names])
                shap_values = explainer(X[er_model.feature_names])
                # compute mean absolute shap per feature
                mean_abs = np.abs(shap_values.values).mean(axis=0)
                feat_imp = sorted(
                    zip(er_model.feature_names, mean_abs.tolist()), key=lambda x: x[1], reverse=True
                )[:max_display]
                return {"method": "shap", "feature_importances": feat_imp}
            except Exception:
                logger.exception("SHAP explain failed, falling back to importances")
        # fallback: try model feature_importances_ or coefficients
        if hasattr(er_model.model, "feature_importances_"):
            imps = list(er_model.model.feature_importances_)
            feat_imp = sorted(zip(er_model.feature_names, imps), key=lambda x: x[1], reverse=True)[:max_display]
            return {"method": "feature_importances", "feature_importances": feat_imp}
        if hasattr(er_model.model, "coef_"):
            coefs = np.ravel(getattr(er_model.model, "coef_", []))
            feat_imp = sorted(zip(er_model.feature_names, coefs.tolist()), key=lambda x: abs(x[1]), reverse=True)[:max_display]
            return {"method": "coefficients", "feature_importances": feat_imp}
        # last resort: permutation importance (slow) if sklearn available
        if _HAS_SKLEARN:
            try:
                from sklearn.inspection import permutation_importance  # type: ignore

                res = permutation_importance(er_model.model, X[er_model.feature_names], np.zeros(len(X)), n_repeats=5, random_state=0)
                imps = res.importances_mean
                feat_imp = sorted(zip(er_model.feature_names, imps.tolist()), key=lambda x: x[1], reverse=True)[:max_display]
                return {"method": "permutation", "feature_importances": feat_imp}
            except Exception:
                logger.exception("Permutation importance failed")
        return {"method": "none", "feature_importances": []}
    except Exception:
        logger.exception("explain_model_predictions failed")
        return {"method": "error", "feature_importances": []}


# ---------- Cross-sectional ranking & position sizing ----------


def cross_sectional_rank(
    assets: Sequence[str],
    feature_matrix: Dict[str, Dict[str, float]],
    score_key: str = "final_score",
    top_pct: float = 0.05,
) -> List[Tuple[str, float]]:
    """
    Rank assets by a given score key in feature_matrix (mapping asset->features).
    Returns list of (asset, score) sorted desc.
    """
    rows = []
    for a in assets:
        feats = feature_matrix.get(a, {})
        val = float(feats.get(score_key, 0.0))
        rows.append((a, val))
    rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
    # can return full ranking
    return rows_sorted


def kelly_fraction(p: float, b: float) -> float:
    """
    Kelly fraction for binary bet where:
        p = probability of win
        b = win/loss payoff ratio (e.g., if win gains b*stake, lose loses stake)
    Returns fraction (can be >1).
    """
    if p <= 0 or p >= 1 or b <= 0:
        return 0.0
    return max(0.0, (p * (b + 1) - 1) / b)


def compute_position_size(
    *,
    cfg: Any,
    score: float,
    confidence: float,
    annual_vol: float,
    base_size: Optional[float] = None,
    max_single: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute recommended position size based on score/confidence and volatility.

    Logic:
    - base_size from cfg or parameter
    - conviction = score (0..1) * confidence
    - vol_factor = target_vol / annual_vol
    - size = base_size * conviction * vol_factor (clamped)
    - apply kelly adjustment optionally if model probability provided in cfg

    Returns dict: {'size': x, 'reason': '...'}
    """
    base_size = base_size if base_size is not None else float(getattr(cfg, "BASE_POSITION_SIZE", 0.05))
    max_single = max_single if max_single is not None else float(getattr(cfg, "MAX_SINGLE_POSITION", 0.2))
    target_vol = float(getattr(cfg, "TARGET_PORTFOLIO_VOL", 0.08))

    conviction = float(score) * float(confidence)
    if annual_vol is None or annual_vol <= 0:
        vol_factor = 1.0
    else:
        vol_factor = float(target_vol) / float(annual_vol) if annual_vol != 0 else 1.0
    raw = base_size * conviction * vol_factor
    size = float(max(0.0, min(max_single, raw)))
    reason = f"base={base_size:.3f} conviction={conviction:.3f} vol_factor={vol_factor:.3f}"
    return {"size": size, "reason": reason, "conviction": conviction}


# ---------- High-level API: compute final score & recommendation ----------


def combine_scores(
    *,
    cfg: Any,
    symbol: str,
    price_df: pd.DataFrame,
    fundamentals: Optional[Dict[str, Any]] = None,
    events: Optional[Sequence[Dict[str, Any]]] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
    model: Optional[EventReactionModel] = None,
) -> Dict[str, Any]:
    """
    Compute final score (0..10), probabilities, confidence and position recommendation.

    Steps:
    - compute feature vector
    - compute sub-scores (fund, market, event)
    - combine weights from cfg
    - optionally call ML model to get probability / expected return
    - compute confidence (based on sample sizes / model availability)
    - compute position size

    Returns:
        dict with keys: final_score (0..10), components, probability, expected_return, confidence, position
    """
    # build feature vector
    features, feature_names = build_feature_vector(cfg=cfg, symbol=symbol, price_df=price_df, fundamentals=fundamentals, events=events, benchmark_df=benchmark_df)
    # sub-scores
    fund_score = float(features.get("fund_score", 0.0))
    market_score = float(features.get("m_momentum_6m", 0.0) if "m_momentum_6m" in features else 0.0)
    # better market score: combination of momentum and vol
    try:
        vol = features.get("annualized_vol", 0.0)
        momentum = features.get("momentum_6m", features.get("m_momentum_6m", 0.0))
        # scale momentum to 0..1 (heuristic)
        market_score = float(1 / (1 + math.exp(-momentum * 3)))  # sigmoid
    except Exception:
        market_score = 0.5

    event_score = float(features.get("event_score", 0.5))

    # weights
    wf = float(getattr(cfg, "WEIGHT_FUNDAMENTAL", 0.35))
    wm = float(getattr(cfg, "WEIGHT_MARKET", 0.30))
    we = float(getattr(cfg, "WEIGHT_EVENT", 0.35))
    total_w = wf + wm + we
    if total_w <= 0:
        wf, wm, we = 0.33, 0.33, 0.34
        total_w = 1.0
    wf /= total_w
    wm /= total_w
    we /= total_w

    # combined raw score 0..1
    raw_score = wf * fund_score + wm * market_score + we * event_score
    # map to 0..10 scale
    final_score_10 = float(max(0.0, min(10.0, raw_score * 10.0)))

    # confidence: based on number of events and optional model quality
    n_events = int(features.get("evt_n", 0))
    model_available = model is not None
    model_conf = 0.6 if model_available else 0.0
    # confidence increases with n_events and model presence
    conf = min(1.0, 0.2 + min(1.0, n_events / max(1, getattr(cfg, "MIN_ANALOGS_REQUIRED", 30))) * 0.6 + model_conf * 0.2)

    # ML model predictions if available
    prob_up = None
    expected_excess = None
    if model is not None:
        try:
            X = pd.DataFrame([features])
            prob_up = float(model.predict_probs(X)[0])
            expected_excess = float(model.predict_expected_return(X)[0])
        except Exception:
            logger.exception("Model prediction failed; continuing without ML output")
            prob_up = None
            expected_excess = None

    # position sizing
    pos = compute_position_size(cfg=cfg, score=raw_score, confidence=conf, annual_vol=float(features.get("annualized_vol", 0.0)))

    # build explanation
    explain = {
        "weights": {"fund": wf, "market": wm, "event": we},
        "components": {"fund_score": fund_score, "market_score": market_score, "event_score": event_score},
        "features_used": feature_names,
    }

    result = {
        "symbol": symbol,
        "final_score_0_1": raw_score,
        "final_score_10": final_score_10,
        "confidence": conf,
        "prob_up": prob_up,
        "expected_excess": expected_excess,
        "position": pos,
        "explain": explain,
        "n_events": n_events,
        "timestamp": datetime.utcnow().isoformat(),
    }
    return result


# ---------- Utilities / small helpers ----------


def score_batch_assets(
    cfg: Any,
    assets: Sequence[str],
    price_loader: Any,
    fundamentals_loader: Any,
    events_loader: Any,
    benchmark_loader: Any,
    model: Optional[EventReactionModel] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Score a batch of assets.

    - price_loader(symbol) -> pd.DataFrame
    - fundamentals_loader(symbol) -> dict
    - events_loader(symbol) -> list of events
    - benchmark_loader() -> pd.DataFrame

    Returns mapping symbol->result (see combine_scores output)
    """
    results: Dict[str, Dict[str, Any]] = {}
    bench_df = benchmark_loader() if callable(benchmark_loader) else None
    for sym in assets:
        try:
            df = price_loader(sym)
            fund = fundamentals_loader(sym)
            evs = events_loader(sym)
            res = combine_scores(cfg=cfg, symbol=sym, price_df=df, fundamentals=fund, events=evs, benchmark_df=bench_df, model=model)
            results[sym] = res
        except Exception:
            logger.exception("Failed scoring asset %s", sym)
    return results


# ---------- Module-level training/test helpers ----------


def backtest_strategy_on_historical(
    cfg: Any,
    symbol: str,
    price_df: pd.DataFrame,
    signals: Sequence[float],
    rebalancing_period_days: int = 30,
) -> Dict[str, Any]:
    """
    Simple backtest: follow discrete signals (0..1) as fraction of portfolio allocated each period.

    Returns P&L metrics dictionary.
    """
    # naive backtest assuming signals are monthly and price_df daily
    if price_df is None or price_df.empty:
        return {}
    df = price_df.sort_values("timestamp").reset_index(drop=True)
    # resample monthly close
    df_month = df.set_index("timestamp").resample("30D").last().dropna(subset=["adj_close"])
    prices = df_month["adj_close"].values
    if len(prices) < 2 or len(signals) < 1:
        return {}
    # align signals length
    n = min(len(signals), len(prices) - 1)
    returns = []
    for i in range(n):
        pct_alloc = float(signals[i])
        ret = (prices[i + 1] / prices[i] - 1.0) * pct_alloc
        returns.append(ret)
    # metrics
    cumulative = np.prod([1 + r for r in returns]) - 1.0 if returns else 0.0
    ann_ret = (1 + cumulative) ** (252.0 / (len(returns) * 30.0)) - 1.0 if returns else 0.0
    ann_vol = float(np.std(returns, ddof=1) * math.sqrt(252)) if len(returns) > 1 else 0.0
    dd = max_drawdown(list(prices[: n + 1])) or 0.0
    return {"cumulative": cumulative, "annual_return": ann_ret, "annual_vol": ann_vol, "max_drawdown": dd}


# ---------- End of module ----------