import math

import numpy as np
import torch
import torch.nn.functional as F

from langinfo.uid import compute_pm_word_surprisals, compute_uid_profile


def test_uid_uniform_profile():
    widths = np.ones(10)
    surps = np.full(10, 4.5)
    prof = compute_uid_profile(widths, surps)
    assert prof["valid"] is True
    assert abs(prof["mu"] - 0.5) < 1e-12
    assert np.allclose(prof["G"], np.linspace(0.0, 1.0, 21))


def test_uid_early_vs_late_symmetry():
    widths = np.ones(6)
    early = np.array([12.0, 6.0, 3.0, 1.5, 1.0, 0.5])
    late = early[::-1]
    p_early = compute_uid_profile(widths, early)
    p_late = compute_uid_profile(widths, late)

    assert p_early["mu"] < 0.5 < p_late["mu"]
    assert abs((p_early["mu"] + p_late["mu"]) - 1.0) < 1e-12


def test_uid_rejects_short_or_zero_info():
    assert compute_uid_profile([1.0, 1.0], [3.0, 4.0])["valid"] is False
    assert compute_uid_profile([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])["valid"] is False


def test_pm_word_surprisals_telescope_to_sentence_score():
    torch.manual_seed(7)
    vocab = 32
    bow_ids = {4, 5, 6, 7, 8}
    b_bow = sorted(bow_ids | {3})
    target_ids = [4, 12, 13, 5, 14, 6, 15]  # 3 words
    span_logits = torch.randn(len(target_ids) + 1, vocab)

    # Without extended punctuation set at the final boundary, word scores must telescope
    word_surps = compute_pm_word_surprisals(span_logits, target_ids, bow_ids, b_bow, b_eow_ids=None)

    lp = F.log_softmax(span_logits, dim=-1)
    tgt_t = torch.tensor(target_ids, dtype=torch.long)
    naive = (-lp[: len(target_ids)].gather(1, tgt_t[:, None]).squeeze(1) / math.log(2)).sum().item()
    b_t = torch.tensor(b_bow, dtype=torch.long)
    left = torch.logsumexp(lp[0, b_t], dim=-1).item() / math.log(2)
    right = -torch.logsumexp(lp[-1, b_t], dim=-1).item() / math.log(2)
    sent_pm = naive + left + right

    assert abs(float(word_surps.sum()) - sent_pm) < 1e-5
