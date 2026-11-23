import numpy as np
import pandas as pd

from langinfo.stats import cluster_boot_mean, cluster_boot_uid
from langinfo.uid import DEFAULT_GRID, compute_uid_profile


def test_cluster_boot_mean_basic_and_reproducible():
    vals = [1.0, 1.2, 2.0, 2.1, 3.0, 2.9]
    arts = [0, 0, 1, 1, 2, 2]
    r1 = cluster_boot_mean(vals, arts, n_boot=500, seed=123)
    r2 = cluster_boot_mean(vals, arts, n_boot=500, seed=123)

    assert r1 == r2
    assert r1["lo"] <= r1["estimate"] <= r1["hi"]
    assert r1["n_clusters"] == 3


def test_cluster_boot_uid_shapes():
    rows = []
    for art in range(10):
        prof = compute_uid_profile(np.ones(5), np.array([2.0, 2.0, 2.0, 2.0, 2.0]))
        row = {"art": art, "mu": prof["mu"]}
        for idx, g_val in enumerate(prof["G"]):
            row[f"G_{idx:02d}"] = g_val
        rows.append(row)
    res = cluster_boot_uid(pd.DataFrame(rows), grid=DEFAULT_GRID, n_boot=200, seed=0)

    assert abs(res["mu_est"] - 0.5) < 1e-12
    assert res["G_est"].shape == (len(DEFAULT_GRID),)
    assert np.allclose(res["diff_est"], 0.0, atol=1e-12)
