"""Direction model with purged walk-forward validation.

The model estimates P(up): the probability that price reaches the upper
triple barrier first (or ends the horizon higher). It is a validated stack of
three components pooled across all pairs:

* TrendFactor  - one-parameter time-series momentum composite (robust prior)
* gradient-boosted trees - non-linear interactions (regime x momentum, ...)
* logistic regression - regularised linear model on every feature

FX labels are noisy and non-stationary, so free-form models often learn
period-specific patterns. Each fit therefore holds out the most recent part
of its training window, scores every component there, weights components by
their validated skill (AUC above 0.5) and calibrates the blend. When no
component shows skill the model outputs 0.5, i.e. "no edge".

Walk-forward: the timeline is cut into consecutive test blocks. Each block is
predicted by a model trained only on samples whose label was fully resolved
before the block starts (purging), so overlapping labels cannot leak future
information into training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig


TREND_FEATURES = ["mom_24", "mom_120", "mom_480", "ema50_dist", "ema200_dist", "ema50_slope",
                  "h4_ema20_dist", "d1_ema20_dist", "strength_spread_day", "strength_spread_week",
                  "mom_5", "mom_20", "mom_60", "w1_ema20_dist"]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class TrendFactor:
    """Average of standardised trend features; only its sign/scale is fitted."""

    def __init__(self, feature_names: list[str]):
        self.idx = [i for i, f in enumerate(feature_names) if f in TREND_FEATURES]

    def _score(self, X: np.ndarray) -> np.ndarray:
        z = np.clip((X[:, self.idx] - self.mu) / self.sd, -4, 4)
        return np.nan_to_num(z).mean(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrendFactor":
        if not self.idx:
            raise ValueError("no trend features available")
        self.mu = np.nanmean(X[:, self.idx], axis=0)
        self.sd = np.nanstd(X[:, self.idx], axis=0) + 1e-12
        self.lr = LogisticRegression(C=1.0).fit(self._score(X), y)
        return self

    def logit(self, X: np.ndarray) -> np.ndarray:
        return self.lr.decision_function(self._score(X))


class _Sklearn:
    def __init__(self, estimator):
        self.est = estimator

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def logit(self, X):
        return _logit(self.est.predict_proba(X)[:, 1])


class StackedModel:
    COMPONENTS = ("trend", "boosting", "logistic")

    def __init__(self, feature_names: list[str], random_state: int = 42, validation_frac: float = 0.25):
        self.names = feature_names
        self.random_state = random_state
        self.validation_frac = validation_frac
        self.weights = np.zeros(len(self.COMPONENTS))
        self.validation_auc = np.full(len(self.COMPONENTS), np.nan)

    def _components(self) -> list:
        # Heavily regularised: overlapping labels make the effective sample
        # size far smaller than the row count.
        gbm = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=100, max_depth=3, min_samples_leaf=1000,
            l2_regularization=5.0, max_features=0.5, random_state=self.random_state)
        logistic = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(C=0.01, max_iter=1000))
        return [TrendFactor(self.names), _Sklearn(gbm), _Sklearn(logistic)]

    def fit(self, X: np.ndarray, y: np.ndarray, time: np.ndarray, label_end: np.ndarray) -> "StackedModel":
        """Rows must be in time order. `label_end` is when each row's label resolved."""
        n = len(y)
        v0 = int(n * (1 - self.validation_frac))
        fit_rows = np.flatnonzero(label_end[:v0] < time[v0])  # purge labels overlapping validation
        val = slice(v0, n)

        comps = [c.fit(X[fit_rows], y[fit_rows]) for c in self._components()]
        z_fit = np.column_stack([c.logit(X[fit_rows]) for c in comps])
        z_val = np.column_stack([c.logit(X[val]) for c in comps]) / (z_fit.std(axis=0) + 1e-12)
        y_val = y[val]
        if len(np.unique(y_val)) == 2:
            self.validation_auc = np.array([roc_auc_score(y_val, z) for z in z_val.T])
        skill = np.clip(np.nan_to_num(self.validation_auc - 0.5), 0, None)
        if skill.sum() == 0:
            self.weights = np.zeros(len(comps))
            return self
        self.weights = skill / skill.sum()
        self.calibrator = LogisticRegression(C=1.0).fit((z_val @ self.weights)[:, None], y_val)

        self.components = [c.fit(X, y) for c in self._components()]
        self.scale = np.column_stack([c.logit(X) for c in self.components]).std(axis=0) + 1e-12
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights.sum() == 0:
            return np.full(len(X), 0.5)
        z = np.column_stack([c.logit(X) for c in self.components]) / self.scale
        return self.calibrator.predict_proba((z @ self.weights)[:, None])[:, 1]


class OOSCalibrator:
    """Platt scaling fitted on the walk-forward out-of-sample predictions.

    The stack's own calibration uses a short validation window and tends to be
    over-confident, so reported probabilities are recalibrated on the full OOS
    history. Signals and rankings use the raw score (recalibration is monotone
    and would not change them). A non-positive slope (no demonstrated skill)
    maps everything to 0.5 rather than inverting signals.
    """

    def __init__(self, fallback_shrink: float = 0.5):
        self.fallback_shrink = fallback_shrink
        self.lr: LogisticRegression | None = None

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "OOSCalibrator":
        self.lr = LogisticRegression(C=1.0).fit(_logit(p_raw)[:, None], y)
        return self

    @property
    def slope(self) -> float:
        return float(self.lr.coef_[0, 0]) if self.lr is not None else np.nan

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        if self.lr is None:
            return 0.5 + self.fallback_shrink * (np.asarray(p_raw) - 0.5)
        if self.slope <= 0:
            return np.full(len(p_raw), 0.5)
        return self.lr.predict_proba(_logit(p_raw)[:, None])[:, 1]


@dataclass
class WalkForwardResult:
    oos_proba: pd.Series
    folds: pd.DataFrame
    calibrator: OOSCalibrator = field(default_factory=OOSCalibrator)
    last_model: StackedModel | None = None
    last_test_mask: np.ndarray | None = field(default=None, repr=False)


def _naive(values) -> np.ndarray:
    """datetime64 array in UTC without timezone (fast vectorised comparisons)."""
    s = pd.Series(values)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.to_numpy()


def _train_rows(dataset: pd.DataFrame, cutoff: pd.Timestamp | None, stride: int) -> np.ndarray:
    ok = dataset["target"].notna().to_numpy()
    if cutoff is not None:
        ok = ok & (dataset["label_end_time"] < cutoff).to_numpy()
    rows = np.flatnonzero(ok)
    return rows[::stride]


def walk_forward(dataset: pd.DataFrame, feature_cols: list[str], cfg: ModelConfig,
                 log=print, min_calibration_rows: int = 5000) -> WalkForwardResult:
    X = dataset[feature_cols].to_numpy(dtype=np.float32)
    y = dataset["target"].to_numpy()
    times = dataset["time"]
    t = _naive(times)
    t_end = _naive(dataset["label_end_time"])
    labelled = ~np.isnan(y)
    t0, t1 = times.iloc[0], times.iloc[-1]
    step = pd.Timedelta(cfg.retrain_every)
    edges = list(pd.date_range(t0 + (t1 - t0) * cfg.initial_train_frac, t1, freq=step))
    if len(edges) > 1 and t1 - edges[-1] < step / 2:
        edges.pop()  # fold the final sliver into the previous test block
    edges.append(t1 + pd.Timedelta(seconds=1))

    proba = np.full(len(dataset), np.nan)
    folds, last_model, last_mask = [], None, None
    for start, end in zip(edges[:-1], edges[1:]):
        train = _train_rows(dataset, start, cfg.train_stride)
        test = ((times >= start) & (times < end)).to_numpy()
        if len(train) < cfg.min_train_rows or not test.any():
            continue
        model = StackedModel(feature_cols, cfg.random_state).fit(X[train], y[train], t[train], t_end[train])
        proba[test] = model.predict_proba(X[test])
        scored = test & labelled
        auc = roc_auc_score(y[scored], proba[scored]) if len(np.unique(y[scored])) == 2 else np.nan
        weights = {k: round(float(w), 2) for k, w in zip(StackedModel.COMPONENTS, model.weights)}
        folds.append({"test_start": start, "test_end": min(end, t1), "train_rows": len(train),
                      "test_rows": int(test.sum()), "auc": auc, **{f"w_{k}": v for k, v in weights.items()}})
        log(f"[model] fold {len(folds):2d}: test {start:%Y-%m-%d} -> {min(end, t1):%Y-%m-%d}  "
            f"train={len(train):,}  AUC={auc:.3f}  weights={weights}")
        if scored.sum() >= 500:
            last_model, last_mask = model, scored

    calibrator = OOSCalibrator()
    history = ~np.isnan(proba) & labelled
    if history.sum() >= min_calibration_rows:
        calibrator.fit(proba[history], y[history])
    return WalkForwardResult(pd.Series(proba, index=dataset.index), pd.DataFrame(folds), calibrator,
                             last_model, last_mask)


def fit_final(dataset: pd.DataFrame, feature_cols: list[str], cfg: ModelConfig) -> StackedModel:
    """Model trained on every resolved label, used for today's predictions."""
    rows = _train_rows(dataset, None, cfg.train_stride)
    X = dataset[feature_cols].to_numpy(dtype=np.float32)
    return StackedModel(feature_cols, cfg.random_state).fit(
        X[rows], dataset["target"].to_numpy()[rows], _naive(dataset["time"])[rows],
        _naive(dataset["label_end_time"])[rows])


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def score(y: np.ndarray, p: np.ndarray, threshold: float = 0.04) -> dict:
    mask = ~(np.isnan(y) | np.isnan(p))
    y, p = y[mask], p[mask]
    if len(y) < 50 or len(np.unique(y)) < 2:
        return {"n": int(len(y))}
    confident = np.abs(p - 0.5) > threshold
    hit = ((p > 0.5) == (y == 1))
    base = y.mean()
    return {
        "n": int(len(y)),
        "base_rate": float(base),
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(hit.mean()),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "log_loss_baseline": float(log_loss(y, np.full_like(p, base))),
        "brier": float(brier_score_loss(y, p)),
        "signal_share": float(confident.mean()),
        "signal_hit_rate": float(hit[confident].mean()) if confident.any() else np.nan,
    }


def score_by(dataset: pd.DataFrame, proba_col: str, group: str, threshold: float = 0.04) -> pd.DataFrame:
    rows = []
    for key, g in dataset.groupby(group, sort=True):
        s = score(g["target"].to_numpy(), g[proba_col].to_numpy(), threshold)
        if "auc" in s:
            rows.append({group: key, **s})
    return pd.DataFrame(rows).set_index(group) if rows else pd.DataFrame()


def calibration_table(y: np.ndarray, p: np.ndarray, bins=(0, .4, .45, .48, .52, .55, .6, 1.0)) -> pd.DataFrame:
    mask = ~(np.isnan(y) | np.isnan(p))
    df = pd.DataFrame({"y": y[mask], "p": p[mask]})
    df["bucket"] = pd.cut(df["p"], bins=list(bins), include_lowest=True)
    out = df.groupby("bucket", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), realised_up=("y", "mean"))
    out.index = out.index.astype(str)
    return out


def permutation_importance(model: StackedModel, X: np.ndarray, y: np.ndarray, names: list[str],
                           n_repeats: int = 3, max_rows: int = 20_000, seed: int = 0) -> pd.Series:
    """Drop in out-of-sample AUC when each feature is shuffled."""
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]
    if len(y) > max_rows:
        keep = rng.choice(len(y), max_rows, replace=False)
        X, y = X[keep], y[keep]
    if len(np.unique(y)) < 2:
        return pd.Series(dtype=float)
    base = roc_auc_score(y, model.predict_proba(X))
    drops = {}
    for j, name in enumerate(names):
        saved = X[:, j].copy()
        scores = []
        for _ in range(n_repeats):
            X[:, j] = rng.permutation(saved)
            scores.append(roc_auc_score(y, model.predict_proba(X)))
        X[:, j] = saved
        drops[name] = base - float(np.mean(scores))
    return pd.Series(drops).sort_values(ascending=False)
