"""Lightweight CPU example demonstrating the full langinfo workflow.

Builds a small MonoT model, scores synthetic multi-article token sequences
with the Pimentel & Meister word-boundary correction, aggregates UID curves
via article-level cluster bootstrap, and exports a summary figure.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from langinfo import (
    DEFAULT_GRID,
    MonoT,
    cluster_boot_mean,
    cluster_boot_uid,
    compute_pm_word_surprisals,
    compute_uid_profile,
    strip_final_punct,
)
from langinfo.plotting import plot_uid_panels


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(42)

    # 1. Verify punctuation cleaning across scripts
    sample_sentences = {
        "fr": "La ville de Lyon est situee au confluent du Rhone et de la Saone.",
        "ja": "これはテスト文です。",
        "hi": "यह एक परीक्षण वाक्य है।",
        "ar": "هل هذا مثال تجريبي؟",
    }
    print("Trailing punctuation cleanup:")
    for lang, sent in sample_sentences.items():
        print(f"  [{lang}] {sent}  ->  {strip_final_punct(sent)}")

    # 2. Forward pass and Pimentel & Meister word-level scoring on a tiny CPU MonoT
    vocab_size = 128
    bow_ids = set(range(4, 64))
    b_bow = sorted(bow_ids | {3})
    b_eow = sorted(bow_ids | {3, 65, 66})  # include two punctuation token IDs

    model = MonoT(vocab_size=vocab_size, d_model=64, n_layer=2, n_head=4, block_size=64, dropout=0.0)
    model.eval()

    # Build a synthetic context + target sequence of 8 words (14 subwords)
    seq = [2, 10, 80, 12, 91, 14, 77, 16, 18, 95, 20, 22, 88, 24, 26, 72]
    m_toks = 12
    with torch.no_grad():
        logits = model(torch.tensor([seq], dtype=torch.long))[0]
    span_logits = logits[len(seq) - m_toks - 1 : len(seq)]
    target_ids = seq[-m_toks:]

    word_surps = compute_pm_word_surprisals(
        span_logits=span_logits,
        target_ids=target_ids,
        bow_ids=bow_ids,
        b_bow_ids=b_bow,
        b_eow_ids=b_eow,
    )
    single_prof = compute_uid_profile(np.ones(len(word_surps)), word_surps)
    print(
        f"\nSingle-sentence MonoT CPU check: {single_prof['n_units']} words, "
        f"I_tot = {single_prof['i_tot']:.2f} bits, mu = {single_prof['mu']:.4f}"
    )

    # 3. Synthetic multi-article UID benchmark (30 articles x 3 sentences per language)
    # Emulates early-peaking (zh), near-uniform (en), and late-peaking (fi) slopes
    slopes = {"zh": -0.35, "en": 0.05, "fi": 0.35}
    item_rows = []
    for lang, slope in slopes.items():
        for art_id in range(30):
            art_shift = rng.normal(0.0, 0.08)
            for s_idx in range(3):
                n_words = int(rng.integers(10, 22))
                pos = np.linspace(-0.5, 0.5, n_words)
                surps = np.clip(4.0 + (slope + art_shift) * pos + rng.normal(0.0, 0.4, n_words), 0.2, None)
                prof = compute_uid_profile(np.ones(n_words), surps)
                row = {"lang": lang, "art": art_id, "item": art_id * 3 + s_idx, "mu": prof["mu"]}
                for g_i, g_val in enumerate(prof["G"]):
                    row[f"G_{g_i:02d}"] = float(g_val)
                item_rows.append(row)

    df_items = pd.DataFrame(item_rows)

    g_rows, mu_rows = [], []
    for lang in slopes:
        sub = df_items[df_items["lang"] == lang]
        boot = cluster_boot_uid(sub, grid=DEFAULT_GRID, n_boot=1000, seed=2026)
        mu_rows.append({"lang": lang, "mu_est": boot["mu_est"], "mu_lo": boot["mu_lo"], "mu_hi": boot["mu_hi"]})
        for idx, x_val in enumerate(DEFAULT_GRID):
            g_rows.append({
                "lang": lang,
                "x": float(x_val),
                "G_est": float(boot["G_est"][idx]),
                "G_lo": float(boot["G_lo"][idx]),
                "G_hi": float(boot["G_hi"][idx]),
                "diff_est": float(boot["diff_est"][idx]),
                "diff_lo": float(boot["diff_lo"][idx]),
                "diff_hi": float(boot["diff_hi"][idx]),
            })

    df_g = pd.DataFrame(g_rows)
    df_mu = pd.DataFrame(mu_rows)

    # Paired bootstrap contrast: FI vs EN
    en_sub = df_items[df_items["lang"] == "en"].set_index("item")
    fi_sub = df_items[df_items["lang"] == "fi"].set_index("item")
    contrast = cluster_boot_mean(
        (fi_sub["mu"] - en_sub["mu"]).to_numpy(),
        en_sub["art"].to_numpy(),
        n_boot=1000,
        seed=2026,
    )

    print("\nArticle-level bootstrap summary (synthetic demo):")
    print(df_mu.round(4).to_string(index=False))
    print(
        f"Paired contrast mu(fi) - mu(en): {contrast['estimate']:+.4f} "
        f"[{contrast['lo']:+.4f}, {contrast['hi']:+.4f}], p = {contrast['p_boot_2sided']:.4f}"
    )

    out_fig = Path(__file__).resolve().parent / "demo_uid_profile.png"
    plot_uid_panels(df_g, df_mu, title="Demo UID profile (synthetic multi-article run)", save_path=out_fig)
    print(f"\nSaved figure to: {out_fig}")


if __name__ == "__main__":
    main()
