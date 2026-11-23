"""Intra-sentence Uniform Information Density (UID) profiles and Pimentel & Meister scoring."""

from __future__ import annotations

import math
from typing import Sequence, Set

import numpy as np
import torch
import torch.nn.functional as F

from .boundaries import split_into_word_spans

DEFAULT_GRID = np.linspace(0.0, 1.0, 21)


def compute_pm_word_surprisals(
    span_logits: torch.Tensor,
    target_ids: Sequence[int],
    bow_ids: Set[int],
    b_bow_ids: Sequence[int],
    b_eow_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Compute word-level surprisal (in bits) with Pimentel & Meister (2024) correction.

    For each word ``w_t`` spanning subwords ``[a, b)``, applies:
        s_PM(w_t) = s_naive(w_t) + 1[u_{t,1} in V_bow] log2 P(B | w_{<t}) - log2 P(B | w_{<=t})

    When ``b_eow_ids`` is provided, it replaces ``b_bow_ids`` at the final closing
    position so that stripping sentence-final punctuation does not inflate the
    last word's surprisal.

    Parameters
    ----------
    span_logits : torch.Tensor
        Logits of shape ``(m + 1, V)`` covering the ``m`` target predictions plus
        the closing position after the last target token.
    target_ids : Sequence[int]
        Target subword token IDs of length ``m``.
    bow_ids : set[int]
        Beginning-of-word token IDs (``V_bow``).
    b_bow_ids : Sequence[int]
        Boundary set ``V_bow U {EOS}``.
    b_eow_ids : Sequence[int], optional
        Extended end-of-word set ``V_bow U {EOS} U V_punct`` for the final word.

    Returns
    -------
    np.ndarray
        Array of corrected word-level surprisals in bits.
    """
    m = len(target_ids)
    if span_logits.shape[0] != m + 1:
        raise ValueError(f"Expected span_logits with {m + 1} rows, got {span_logits.shape[0]}.")

    device = span_logits.device
    lp_all = F.log_softmax(span_logits.float(), dim=-1)
    tgt_t = torch.tensor(list(target_ids), dtype=torch.long, device=device)
    surp = (-lp_all[:m].gather(1, tgt_t[:, None]).squeeze(1) / math.log(2)).cpu().numpy()

    b_bow_t = torch.tensor(list(b_bow_ids), dtype=torch.long, device=device)
    log2_pb_bow = (torch.logsumexp(lp_all[:, b_bow_t], dim=-1) / math.log(2)).cpu().numpy()

    if b_eow_ids is not None:
        b_eow_t = torch.tensor(list(b_eow_ids), dtype=torch.long, device=device)
        log2_pb_end = float(torch.logsumexp(lp_all[m, b_eow_t], dim=-1).item() / math.log(2))
    else:
        log2_pb_end = float(log2_pb_bow[m])

    spans = split_into_word_spans(target_ids, bow_ids)
    word_surps = np.empty(len(spans), dtype=np.float64)
    for idx_w, (a, b) in enumerate(spans):
        s_naive = float(surp[a:b].sum())
        left_corr = float(log2_pb_bow[a]) if (target_ids[a] in bow_ids) else 0.0
        right_corr = -log2_pb_end if (idx_w == len(spans) - 1) else float(-log2_pb_bow[b])
        word_surps[idx_w] = s_naive + left_corr + right_corr
    return word_surps


def compute_uid_profile(
    unit_widths: Sequence[float] | np.ndarray,
    unit_surps: Sequence[float] | np.ndarray,
    grid: np.ndarray = DEFAULT_GRID,
) -> dict:
    """Compute normalized cumulative information curve G(x) and center of mass mu.

    The center of mass ``mu = 1 - integral_0^1 G(x) dx`` equals the midpoint
    barycenter of unit positions weighted by ``s_t / I_tot``. Under perfectly
    uniform information density, ``G(x) = x`` and ``mu = 0.5``.

    Parameters
    ----------
    unit_widths : array-like of float
        Width of each unit along the sentence axis (ones for words; character
        counts for scripts without spaces).
    unit_surps : array-like of float
        Information content (surprisal in bits) of each unit.
    grid : np.ndarray, optional
        Common normalized abscissa grid in ``[0, 1]``.

    Returns
    -------
    dict
        Dictionary with keys ``valid``, ``n_units``, ``i_tot``, ``G``, and ``mu``.
    """
    surps = np.clip(np.asarray(unit_surps, dtype=np.float64), 0.0, None)
    widths = np.asarray(unit_widths, dtype=np.float64)

    if surps.shape != widths.shape:
        raise ValueError("unit_widths and unit_surps must have identical shapes.")

    n_units = int(len(surps))
    i_tot = float(surps.sum())
    w_tot = float(widths.sum())
    if n_units < 3 or i_tot <= 0.0 or w_tot <= 0.0 or np.any(widths < 0.0):
        return {"valid": False, "n_units": n_units, "i_tot": i_tot, "G": None, "mu": float("nan")}

    x_pts = np.concatenate([[0.0], np.cumsum(widths) / w_tot])
    y_pts = np.concatenate([[0.0], np.cumsum(surps) / i_tot])
    x_pts[-1] = 1.0
    y_pts[-1] = 1.0

    g_grid = np.interp(grid, x_pts, y_pts)
    g_grid[0] = 0.0
    g_grid[-1] = 1.0

    # Exact trapezoidal integration over the uninterpolated step vertices
    area = float(np.sum(0.5 * (y_pts[:-1] + y_pts[1:]) * np.diff(x_pts)))
    mu = 1.0 - area
    return {"valid": True, "n_units": n_units, "i_tot": i_tot, "G": g_grid, "mu": mu}
