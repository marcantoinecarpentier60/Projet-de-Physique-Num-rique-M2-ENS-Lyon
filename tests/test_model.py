import torch

from langinfo.model import MonoT, generate_tokens


def test_monot_shape_and_weight_tying():
    model = MonoT(vocab_size=64, d_model=32, n_layer=2, n_head=4, block_size=16, dropout=0.0)
    assert model.head.weight is model.wte.weight
    x = torch.randint(0, 64, (2, 10))
    logits = model(x)
    assert logits.shape == (2, 10, 64)


def test_monot_causal_masking():
    torch.manual_seed(0)
    model = MonoT(vocab_size=64, d_model=32, n_layer=2, n_head=4, block_size=16, dropout=0.0)
    model.eval()
    x1 = torch.tensor([[2, 10, 11, 12, 13, 14]], dtype=torch.long)
    x2 = x1.clone()
    x2[0, -1] = 25  # modify only the last token

    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)
    # Predictions at positions 0..4 must not depend on token 5
    assert torch.allclose(y1[:, :-1, :], y2[:, :-1, :], atol=1e-6)


def test_generate_tokens_respects_length_and_forbidden():
    model = MonoT(vocab_size=64, d_model=32, n_layer=1, n_head=4, block_size=16, dropout=0.0)
    out = generate_tokens(model, prompt_ids=[2, 10, 11], max_new_tokens=6, temperature=0.0)
    assert len(out) <= 6
    assert all(tok not in (0, 1, 2) for tok in out)
