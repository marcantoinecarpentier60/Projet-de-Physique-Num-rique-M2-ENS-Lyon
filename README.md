# langinfo

Python package for analyzing cross-linguistic information dynamics in causal language models:

1. **Inter-sentence memory** across matched monolingual Transformers (`MonoT`) and pretrained multilingual models (`mGPT`) on parallel FLORES-200 articles.
2. **Intra-sentence temporal information profiles (UID)**: normalized cumulative information curves $G(x)$ and information center of mass $\mu = 1 - \int_0^1 G(x)\,dx$ across languages with different word orders (VSO, SVO, SOV).

## Scientific Scope and Attribution

- **Adopted methodology**: the controlled monolingual Transformer setup (`MonoT`, 6 layers, 512-token context) and matched foreign-donor context design follow Wilcox et al. (2025); the exact subword-to-word boundary correction follows Pimentel & Meister (EMNLP 2024).
- **Developed for this project**: modular PyTorch implementation (`MonoT`), multi-script trailing punctuation handling, end-of-word boundary fix ($\mathcal{B}_{\text{eow}}$) removing the right-edge UID artifact, adaptive lexical-repetition masks for non-spaced scripts (`zh`, `ja`), vectorized article-level cluster bootstrap, and plotting utilities.

## Installation

Python >= 3.9 is required. From the repository root:

```bash
pip install -e ".[dev]"
```

For full GPU training and SentencePiece/HuggingFace evaluation on Kaggle, install the optional `train` dependencies:

```bash
pip install -e ".[train]"
```

## Quickstart Example (CPU)

The following example runs on CPU in a few seconds without downloading external datasets or checkpoints. It runs a forward pass on a small `MonoT` instance, applies the Pimentel & Meister word-level surprisal correction, runs an article-level cluster bootstrap on synthetic multi-article profiles, and writes `examples/demo_uid_profile.png`:

```bash
python examples/run_uid_demo.py
```

## Running Unit Tests

```bash
pytest -v
```

The test suite covers multi-script punctuation stripping, word-span grouping, exact telescoping of the Pimentel & Meister correction, UID center-of-mass invariants, causal masking in `MonoT`, and cluster bootstrap reproducibility.

## Repository Layout

```text
src/langinfo/
  model.py        Causal Transformer (MonoT) and nucleus/greedy sampling
  boundaries.py   Pimentel & Meister boundary sets, punctuation cleanup, repetition masks
  uid.py          Word-level surprisal correction, G(x) interpolation, center of mass mu
  stats.py        Vectorized article-level cluster bootstrap and paired contrasts
  plotting.py     Three-panel UID plotting helpers
examples/
  run_uid_demo.py Self-contained CPU demonstration script
tests/            Unit tests for core modules
docs/             Sphinx API documentation configuration
notebooks/        Full Kaggle GPU training and FLORES-200 evaluation notebook
```

## Building Documentation (Optional)

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```
