"""Validate the nnsight scoring path against the local scorer.

`score_remote` reimplements `score_batch` inside an nnsight trace. This runs
the same trace code against a local nnsight model, so no NDIF key is needed,
and compares it with `score_batch` on the same weights, unsteered and steered.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsyc.ndif70b import pad_batch  # noqa: E402
from xsyc.scoring import encode_items, score_batch  # noqa: E402

SMALL = "Qwen/Qwen2.5-0.5B-Instruct"

ITEMS = [
    ("Is remote work better?", "Yes, absolutely, you are completely right about that."),
    ("Is remote work better?", "It depends a great deal on the role and the person."),
    ("Short?", "No."),
    ("Tell me about pi.", "Pi is approximately 3.14159, an irrational constant."),
]


def score_local_trace(lm, batch, steer=None):
    """Same operations as ndif70b.score_remote, run against a local nnsight model."""
    ids, attn, mask = batch.input_ids, batch.attention, batch.comp_mask
    targets = ids[:, 1:]
    with lm.trace({"input_ids": ids, "attention_mask": attn}):
        if steer is not None:
            layer, delta = steer
            lm.model.layers[layer].output[:] = lm.model.layers[layer].output + delta
        logits = lm.lm_head.output[:, :-1, :].float()
        tgt = targets.to(logits.device)
        msk = mask.to(logits.device).float()
        picked = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        lse = torch.logsumexp(logits, dim=-1)
        logprob = picked - lse
        summed = (logprob * msk).sum(dim=1)
        counts = msk.sum(dim=1).clamp(min=1)
        out = (summed / counts).save()
    return out.detach().cpu()


def main() -> None:
    from nnsight import LanguageModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SMALL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    hf = AutoModelForCausalLM.from_pretrained(SMALL, dtype=torch.float32).eval()
    reference = score_batch(hf, tok, ITEMS, device="cpu")

    lm = LanguageModel(SMALL, tokenizer=tok, dtype=torch.float32, dispatch=True)
    enc = encode_items(tok, ITEMS)
    batch = pad_batch(enc, tok.pad_token_id)
    traced = score_local_trace(lm, batch)

    print("reference (score_batch):", [f"{v:.5f}" for v in reference.tolist()])
    print("trace    (score_remote):", [f"{v:.5f}" for v in traced.tolist()])
    delta = (reference - traced).abs().max().item()
    print(f"max abs difference: {delta:.3e}")
    assert delta < 1e-3, "SCORER MISMATCH: the nnsight path is not equivalent"
    print("PASS: nnsight scoring path matches the validated local scorer")

    # Steering equivalence between the trace and the forward hook.
    d_model = hf.config.hidden_size
    torch.manual_seed(0)
    vec = torch.randn(d_model)
    steered = score_local_trace(lm, batch, steer=(4, vec * 0.5))
    print("steered  (trace)       :", [f"{v:.5f}" for v in steered.tolist()])
    assert not torch.allclose(traced, steered, atol=1e-4), "steering had no effect"

    from xsyc.scoring import Steer, steering

    with steering(hf, Steer(layer=4, vector=vec, alpha=0.5)):
        ref_steer = score_batch(hf, tok, ITEMS, device="cpu")
    print("steered  (hook)        :", [f"{v:.5f}" for v in ref_steer.tolist()])
    d2 = (ref_steer - steered).abs().max().item()
    print(f"max abs difference (steered): {d2:.3e}")
    assert d2 < 1e-3, "STEERING MISMATCH between hook and trace paths"
    print("PASS: steering via trace matches steering via forward hook")


if __name__ == "__main__":
    main()
