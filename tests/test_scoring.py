"""CPU-only correctness tests for the scorer, on a tiny random model.

`score_batch`'s two failure modes, an off-by-one in the shift and padding
leaking into the mask, are both silent, so they are tested directly.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsyc.scoring import (  # noqa: E402
    Steer,
    _as_ids,
    build_scored_pair,
    score_batch,
    steering,
)

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


@pytest.fixture(scope="module")
def model_and_tok():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TINY)
    tok.chat_template = CHAT_TEMPLATE
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32).eval()
    return model, tok


ITEMS = [
    ("Is remote work better?", "Yes, absolutely, you are completely right."),
    ("Is remote work better?", "It depends a great deal on the role and the person involved, honestly."),
    ("Short?", "No."),
]


def test_boundary_is_exact(model_and_tok):
    """N must count completion tokens only, with no prefix bleed."""
    _, tok = model_and_tok
    prompt, completion = ITEMS[0]
    ids, n_comp = build_scored_pair(tok, prompt, completion)
    prefix = _as_ids(
        tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
    )
    assert len(ids) == len(prefix) + n_comp
    assert ids[: len(prefix)] == prefix
    assert tok.decode(ids[len(prefix):]) == completion


def test_padding_does_not_change_scores(model_and_tok):
    """Scoring an item alone and inside a batch of differently sized items must agree.

    Padding leaking into the mask, or wrong per-row offsets, would show up here.
    """
    model, tok = model_and_tok
    batched = score_batch(model, tok, ITEMS, device="cpu")
    alone = torch.cat([score_batch(model, tok, [it], device="cpu") for it in ITEMS])
    torch.testing.assert_close(batched, alone, rtol=1e-4, atol=1e-4)


def test_order_invariance(model_and_tok):
    """Reversing batch order must not change any score."""
    model, tok = model_and_tok
    fwd = score_batch(model, tok, ITEMS, device="cpu")
    rev = score_batch(model, tok, ITEMS[::-1], device="cpu")
    torch.testing.assert_close(fwd, rev.flip(0), rtol=1e-4, atol=1e-4)


def test_scores_are_length_normalized(model_and_tok):
    """Scores are per-token means, so they must sit in a plausible range."""
    model, tok = model_and_tok
    scores = score_batch(model, tok, ITEMS, device="cpu")
    assert scores.shape == (3,)
    assert (scores < 0).all(), "log-probs must be negative"
    # A tiny random model is near-uniform over its vocab, so the per-token mean
    # is around -log(vocab); a length-scaled sum would be far lower.
    assert (scores > -20).all(), scores


def test_steering_moves_scores_and_unhooks(model_and_tok):
    """A nonzero vector must change scores; leaving the context must restore them."""
    model, tok = model_and_tok
    base = score_batch(model, tok, ITEMS, device="cpu")

    d_model = model.config.hidden_size
    torch.manual_seed(0)
    vec = torch.randn(d_model)

    with steering(model, Steer(layer=1, vector=vec, alpha=5.0)):
        steered = score_batch(model, tok, ITEMS, device="cpu")
    after = score_batch(model, tok, ITEMS, device="cpu")

    assert not torch.allclose(base, steered, atol=1e-3), "steering had no effect"
    torch.testing.assert_close(base, after, rtol=1e-5, atol=1e-5)


def test_alpha_zero_is_a_noop(model_and_tok):
    model, tok = model_and_tok
    base = score_batch(model, tok, ITEMS, device="cpu")
    vec = torch.randn(model.config.hidden_size)
    with steering(model, Steer(layer=1, vector=vec, alpha=0.0)):
        zero = score_batch(model, tok, ITEMS, device="cpu")
    torch.testing.assert_close(base, zero, rtol=1e-5, atol=1e-5)
