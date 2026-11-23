"""Article-level cluster bootstrap estimators for scalar metrics and UID curves."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .uid import DEFAULT_GRID


def _sample_cluster_weights(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    weights = np.zeros((n_boot, n_clusters), dtype=np.float64)
    np.add.at(weights, (np.arange(n_boot)[:, None], draws), 1.0)
    return weights


def cluster_boot_mean(
    values: Sequence[float] | np.ndarray,
    clusters: Sequence[int] | np.ndarray,
    n_boot: int = 3000,
    seed: int = 2026,
) -> dict:
    """Compute mean, 95% percentile CI, and two-sided p-value via cluster bootstrap.

    Resamples entire articles (clusters) with replacement to account for
    within-article correlation across target sentences.
    """
    vals = np.asarray(values, dtype=np.float64)
    arts = np.asarray(clusters)
    if len(vals) == 0 or len(vals) != len(arts):
        raise ValueError("values and clusters must be non-empty and of equal length.")

    uniq, inv = np.unique(arts, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    sums = np.bincount(inv, weights=vals)

    weights = _sample_cluster_weights(len(uniq), n_boot=n_boot, seed=seed)
    boot = (weights @ sums) / (weights @ counts)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_2sided = float(min(1.0, 2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0))))

    return {
        "estimate": float(vals.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_boot_2sided": p_2sided,
        "n_items": int(len(vals)),
        "n_clusters": int(len(uniq)),
    }


def cluster_boot_uid(
    df: pd.DataFrame,
    grid: np.ndarray = DEFAULT_GRID,
    n_boot: int = 3000,
    seed: int = 2026,
) -> dict:
    """Cluster-bootstrap G(x), G(x) - x, and center of mass mu by article ID.

    Parameters
    ----------
    df : pd.DataFrame
        Item-level dataframe containing columns ``art``, ``mu``, and ``G_00..G_20``.
    grid : np.ndarray, optional
        Normalized sentence-position grid matching the ``G_`` columns.
    n_boot : int, default=3000
        Number of bootstrap replicates.
    seed : int, default=2026
        Random seed for reproducibility.
    """
    g_cols = [f"G_{i:02d}" for i in range(len(grid))]
    arts = df["art"].to_numpy()
    mu_arr = df["mu"].to_numpy(dtype=np.float64)
    g_mat = df[g_cols].to_numpy(dtype=np.float64)

    uniq, inv = np.unique(arts, return_inverse=True)
    n_clusters = len(uniq)
    counts = np.bincount(inv).astype(np.float64)
    mu_sums = np.bincount(inv, weights=mu_arr)
    g_sums = np.zeros((n_clusters, len(grid)), dtype=np.float64)
    np.add.at(g_sums, inv, g_mat)

    weights = _sample_cluster_weights(n_clusters, n_boot=n_boot, seed=seed)
    denom = weights @ counts
    mu_boot = (weights @ mu_sums) / denom
    g_boot = (weights @ g_sums) / denom[:, None]

    mu_lo, mu_hi = np.percentile(mu_boot, [2.5, 97.5])
    g_est = g_mat.mean(axis=0)
    g_lo, g_hi = np.percentile(g_boot, [2.5, 97.5], axis=0)

    return {
        "mu_est": float(mu_arr.mean()),
        "mu_lo": float(mu_lo),
        "mu_hi": float(mu_hi),
        "G_est": g_est,
        "G_lo": g_lo,
        "G_hi": g_hi,
        "diff_est": g_est - grid,
        "diff_lo": g_lo - grid,
        "diff_hi": g_hi - grid,
        "n_items": int(len(df)),
        "n_articles": int(n_clusters),
    }
