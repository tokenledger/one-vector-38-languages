"""Llama-3.1-70B scale check, run remotely on NDIF via nnsight.

`scoring.py` expects a local model it can hook; NDIF executes an intervention
graph remotely, so the same operations are expressed inside an nnsight trace.

Two facts about this deployment, both established by measurement:

1. `layer.output` is a plain (batch, seq, d_model) tensor, not a tuple.
   `layer.output[0]` indexes the batch and silently returns the first item.

2. Logits must stay on the server and must not be materialized twice. A batch
   of 128 sequences at 200 tokens with a 128k vocabulary is roughly 13 GB of
   float32, so log-probs are computed in the trace as x[i] - logsumexp(x) and
   only per-item scalars come back.

Real benchmark items (200 to 400 tokens) score at roughly 3 to 4 sequences per
second, and the batch has to stay small on a shared deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
N_LAYERS = 80
DEPTHS = (0.25, 0.50, 0.75)
LAYERS = [int(round(d * N_LAYERS)) for d in DEPTHS]  # 20, 40, 60


def connect():
    """Return a remote LanguageModel handle, API key taken from the environment."""
    from nnsight import CONFIG, LanguageModel

    key = os.environ.get("NDIF_API_KEY")
    if not key:
        raise RuntimeError("NDIF_API_KEY is not set")
    CONFIG.set_default_api_key(key)
    return LanguageModel(MODEL)


# --- Padded batch construction --------------------------------------------


@dataclass
class Padded:
    """A right-padded batch plus everything needed to mask it after the fact."""

    input_ids: torch.Tensor  # (B, T)
    attention: torch.Tensor  # (B, T)
    comp_mask: torch.Tensor  # (B, T-1) True where a completion token's logprob sits
    n_comp: list[int]


def pad_batch(encoded, pad_id: int) -> Padded:
    """Right-pad encoded (ids, n_completion) pairs and build the scoring mask.

    Mirrors `scoring.score_batch` exactly, including the off-by-one: the
    log-prob of token at absolute position p lives at index p-1 of the shifted
    tensor, because position i predicts position i+1.
    """
    lengths = [len(ids) for ids, _ in encoded]
    n_comp = [n for _, n in encoded]
    width = max(lengths)
    b = len(encoded)

    input_ids = torch.full((b, width), pad_id, dtype=torch.long)
    attention = torch.zeros((b, width), dtype=torch.long)
    for i, (ids, _) in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[i, : len(ids)] = 1

    pos = torch.arange(width - 1).unsqueeze(0)
    starts = torch.tensor([L - n for L, n in zip(lengths, n_comp)]).unsqueeze(1)
    ends = torch.tensor(lengths).unsqueeze(1)
    comp_mask = (pos >= starts - 1) & (pos < ends - 1)

    assert torch.equal(comp_mask.sum(dim=1), torch.tensor(n_comp)), (
        "completion mask does not cover exactly N tokens"
    )
    return Padded(input_ids, attention, comp_mask, n_comp)


# --- Remote scoring --------------------------------------------------------


def with_retry(fn, *args, tries: int = 5, base_wait: float = 8.0, **kwargs):
    """Retry a remote call through transient network failures.

    A multi-hour job over the public internet meets a connection reset at
    least once. A CUDA OOM on the shared GPU is re-raised immediately, since
    the same batch would fail again.
    """
    import time as _time

    last = None
    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - network layer raises many types
            last = exc
            name = type(exc).__name__
            if "OutOfMemory" in name or "OutOfMemory" in str(exc):
                raise
            wait = base_wait * (2**attempt)
            print(f"      {name}, retry {attempt + 1}/{tries} in {wait:.0f}s", flush=True)
            _time.sleep(wait)
    raise RuntimeError(f"remote call failed after {tries} tries") from last


def score_remote(lm, batch: Padded, steer: tuple[int, torch.Tensor] | None = None):
    """Length-normalized mean log-prob per item, computed server-side.

    `steer` is (layer, delta) where delta is already scaled; it is added to the
    residual stream at every position, matching the local `steering` hook.

    Returns a (B,) float tensor. Only these B floats cross the network.
    """
    ids = batch.input_ids
    attn = batch.attention
    mask = batch.comp_mask
    targets = ids[:, 1:]

    with lm.trace({"input_ids": ids, "attention_mask": attn}, remote=True):
        if steer is not None:
            layer, delta = steer
            block = lm.model.layers[layer].output
            block[:] = block + delta.to(block.device).to(block.dtype)
        logits = lm.lm_head.output[:, :-1, :].float()
        # The model is sharded across GPUs, so locally built tensors must move
        # to the device the activations landed on.
        tgt = targets.to(logits.device)
        msk = mask.to(logits.device).float()
        # x[i] - logsumexp(x) avoids a second (B, T, vocab) float32 tensor; a
        # direct log_softmax OOMs the shared deployment. Chunking over the batch
        # is not an option: a Python loop inside an nnsight trace does not run.
        picked = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        lse = torch.logsumexp(logits, dim=-1)
        logprob = picked - lse
        summed = (logprob * msk).sum(dim=1)
        counts = msk.sum(dim=1).clamp(min=1)
        scores = (summed / counts).save()
    return scores.detach().cpu().float()


def collect_remote(lm, batch: Padded, layers: list[int]):
    """Mean residual-stream activation over completion tokens, per item per layer.

    Averages inside the trace so the network carries (B, d_model) rather than
    (B, seq, d_model). Returns {layer: (B, d_model)}.
    """
    # Completion positions in *absolute* terms, one index later than the shifted
    # scoring mask, because activations are not shifted the way logits are.
    act_mask = torch.zeros(batch.input_ids.shape, dtype=torch.bool)
    act_mask[:, 1:] = batch.comp_mask
    weights = act_mask.float()
    denom = weights.sum(dim=1, keepdim=True).clamp(min=1)

    # Unrolled on purpose: a Python `for` loop inside an nnsight 0.7 trace
    # returns an empty dict of saves with no error.
    if len(layers) != 3:
        raise ValueError(f"unrolled for exactly 3 layers, got {len(layers)}")
    la, lb, lc = layers

    with lm.trace({"input_ids": batch.input_ids, "attention_mask": batch.attention}, remote=True):
        ha = lm.model.layers[la].output.float()
        pa = ((ha * weights.to(ha.device).unsqueeze(-1)).sum(dim=1)
              / denom.to(ha.device)).save()
        hb = lm.model.layers[lb].output.float()
        pb = ((hb * weights.to(hb.device).unsqueeze(-1)).sum(dim=1)
              / denom.to(hb.device)).save()
        hc = lm.model.layers[lc].output.float()
        pc = ((hc * weights.to(hc.device).unsqueeze(-1)).sum(dim=1)
              / denom.to(hc.device)).save()

    return {
        la: pa.detach().cpu().float(),
        lb: pb.detach().cpu().float(),
        lc: pc.detach().cpu().float(),
    }


def batches(encoded, max_tokens: int = 7_000, max_items: int = 28):
    """Group indices into batches under a padded-token budget.

    Sorted by true token length so padding waste stays low. Both caps matter:
    high-fertility languages hit the token cap long before the item cap. The
    budget is far smaller than the local one because the deployment is shared:
    peak memory is the upcast (batch x seq x 128256) logits, and 7,000 padded
    tokens keeps that near 3.5 GB.
    """
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    cur: list[int] = []
    width = 0
    for i in order:
        w = max(width, len(encoded[i][0]))
        if cur and (w * (len(cur) + 1) > max_tokens or len(cur) >= max_items):
            yield cur
            cur, width = [i], len(encoded[i][0])
        else:
            cur.append(i)
            width = w
    if cur:
        yield cur
