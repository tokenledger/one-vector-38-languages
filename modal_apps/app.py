"""Modal app: the GPU execution environment for the 7-8B experiments.

Every Llama-3.1-8B and Qwen2.5-7B number in the paper is produced here, in an
image with pinned dependencies. The Llama-3.1-70B check runs separately on
NDIF through `scripts/scale_check.py`.

GPU ladder:
  - L4      + Qwen2.5-0.5B-Instruct : pipeline debugging (`smoke`)
  - A100-40 + the 7-8B models        : all reported results

Every function has an explicit timeout and scaledown_window=60, weights live
on a Volume, and results are checkpointed per language.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

APP = modal.App("xsyc")

HF_CACHE = modal.Volume.from_name("xsyc-hf-cache", create_if_missing=True)
RESULTS = modal.Volume.from_name("xsyc-results", create_if_missing=True)
HF_DIR = "/cache/hf"
OUT_DIR = "/results"

SRC = Path(__file__).resolve().parents[1] / "src"

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",
        "accelerate==1.6.0",
        "huggingface_hub==0.30.2",
        "pandas==2.2.3",
        "pyarrow==19.0.1",
        "scipy==1.15.2",
    )
    # No hf_transfer: it hides the underlying HTTP error, so a gated-repo 403
    # surfaces as an unreadable RuntimeError.
    .env({"HF_HOME": HF_DIR, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(SRC, remote_path="/root/src")
)

# Read from the local environment per invocation; HF_TOKEN must be exported
# when running `modal run`.
HF_SECRET = modal.Secret.from_local_environ(["HF_TOKEN"])

DATASET = "aryashah00/multilingual-sycophancy"
DATA_DIR = f"{HF_DIR}/benchmark"


# --- Fetching -------------------------------------------------------------


@APP.function(
    image=IMAGE,
    volumes={HF_DIR: HF_CACHE},
    secrets=[HF_SECRET],
    timeout=60 * 60,
    scaledown_window=60,
)
def fetch(models: list[str]) -> dict:
    """Pull the benchmark and model weights onto the Volume. CPU only, run once."""
    from huggingface_hub import snapshot_download

    t0 = time.time()
    snapshot_download(DATASET, repo_type="dataset", local_dir=DATA_DIR)
    got = sorted(p.name for p in Path(DATA_DIR).glob("sycophancy_*.jsonl"))

    for m in models:
        snapshot_download(m, ignore_patterns=["*.pth", "original/*"])

    HF_CACHE.commit()
    return {"languages": len(got), "models": models, "seconds": round(time.time() - t0)}


# --- Phase 1: vector extraction and English-only selection ----------------


@APP.function(
    image=IMAGE,
    gpu="A100-40GB",
    volumes={HF_DIR: HF_CACHE, OUT_DIR: RESULTS},
    secrets=[HF_SECRET],
    timeout=60 * 60,
    scaledown_window=60,
)
def extract_vectors(
    model_id: str,
    depths: tuple[float, ...] = (0.25, 0.5, 0.75),
    source: str = "bench",
) -> dict:
    """Build a steering vector from English data only.

    source="bench" -> benchmark English `extract` split; the resistant arm hedges.
    source="fact"  -> authored factual-correction pairs; the resistant arm
                      corrects the user and states the fact.

    Saves, per candidate layer, the raw difference vector and the mean
    residual-stream norm at that layer; the norm is what makes alpha comparable
    across depths (see scoring.scaled_steer).
    """
    import sys

    sys.path.insert(0, "/root/src")
    import torch

    from xsyc import data as D
    from xsyc.scoring import collect_activations, difference_vector

    model, tok = _load_model(model_id)
    n_layers = model.config.num_hidden_layers
    layers = sorted({max(1, min(n_layers - 1, round(d * n_layers))) for d in depths})

    if source == "bench":
        records = D.take(D.load_language("en", DATA_DIR), "extract")
        syc = [(r.prompt, r.sycophantic) for r in records]
        non = [(r.prompt, r.not_sycophantic) for r in records]
    elif source == "fact":
        from xsyc import factpairs as FP

        records = FP.build()
        syc = [(r.prompt, r.sycophantic) for r in records]
        non = [(r.prompt, r.resistant) for r in records]
    else:
        raise ValueError(f"unknown source {source!r}")

    t0 = time.time()
    acts_syc = collect_activations(model, tok, syc, layers)
    acts_non = collect_activations(model, tok, non, layers)

    payload, summary = {}, {}
    for lyr in layers:
        v = difference_vector(acts_non[lyr], acts_syc[lyr])
        ref = float(torch.cat([acts_syc[lyr], acts_non[lyr]]).norm(dim=-1).mean())
        payload[lyr] = {"vector": v, "ref_norm": ref}
        # Cosine near 1 means the difference vector is a small residual of two
        # nearly identical class means.
        cos = torch.nn.functional.cosine_similarity(
            acts_non[lyr].mean(0), acts_syc[lyr].mean(0), dim=0
        )
        summary[lyr] = {
            "v_norm": round(float(v.norm()), 3),
            "ref_norm": round(ref, 3),
            "rel": round(float(v.norm()) / ref, 4),
            "class_mean_cos": round(float(cos), 4),
        }

    short = model_id.split("/")[-1]
    out = Path(OUT_DIR) / "vectors" / f"v_{source}__{short}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model_id, "n_layers": n_layers, "layers": layers, **payload}, out)
    RESULTS.commit()

    return {
        "model": model_id,
        "source": source,
        "n_layers": n_layers,
        "layers": layers,
        "n_pairs": len(records),
        "seconds": round(time.time() - t0),
        "per_layer": summary,
    }


@APP.function(
    image=IMAGE,
    gpu="A100-40GB",
    volumes={HF_DIR: HF_CACHE, OUT_DIR: RESULTS},
    secrets=[HF_SECRET],
    timeout=60 * 90,
    scaledown_window=60,
)
def sweep_english(
    model_id: str,
    alphas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8),
    controls: bool = True,
    source: str = "bench",
) -> list[dict]:
    """Layer x alpha selection on the English `dev` split.

    All selection happens here; nothing downstream re-tunes.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import torch

    from xsyc import data as D
    from xsyc.scoring import (
        encode_items,
        scaled_steer,
        score_batch,
        steering,
        token_budget_for,
        unit,
    )

    model, tok = _load_model(model_id)
    budget = token_budget_for(model)
    short = model_id.split("/")[-1]
    blob = torch.load(Path(OUT_DIR) / "vectors" / f"v_{source}__{short}.pt", weights_only=False)
    layers = blob["layers"]

    records = D.take(D.load_language("en", DATA_DIR), "dev")
    items, arms = [], []
    for r in records:
        items.append((r.prompt, r.sycophantic))
        arms.append("syc")
        items.append((r.prompt, r.not_sycophantic))
        arms.append("nonsyc")
    enc = encode_items(tok, items)
    true_len = [len(ids) for ids, _ in enc]
    batches = list(_batches(items, true_len, budget))
    is_syc = torch.tensor([a == "syc" for a in arms])

    def run(steer) -> dict:
        scores = torch.full((len(items),), float("nan"))
        with steering(model, steer):
            for idx in batches:
                scores[idx] = score_batch(model, tok, encoded=[enc[i] for i in idx])
        assert not torch.isnan(scores).any()
        s, n = scores[is_syc], scores[~is_syc]
        return {
            "margin": round(float((n - s).mean()), 4),
            "syc_rate": round(float((s > n).float().mean()), 4),
        }

    out: list[dict] = []
    base = run(None)
    out.append({"arm": "baseline", "layer": None, "alpha": 0.0, **base})
    print(f"baseline: {base}", flush=True)

    torch.manual_seed(0)
    for lyr in layers:
        v = blob[lyr]["vector"]
        ref = blob[lyr]["ref_norm"]
        for a in alphas:
            if a == 0.0:
                continue
            r = run(scaled_steer(v, lyr, a, ref))
            r["d_margin"] = round(r["margin"] - base["margin"], 4)
            out.append({"arm": f"v_{source}", "layer": lyr, "alpha": a, **r})
            print(f"L{lyr} a={a}: {r}", flush=True)

        if controls:
            # Matched-norm random direction.
            rnd = unit(torch.randn_like(v))
            r = run(scaled_steer(rnd, lyr, 0.2, ref))
            r["d_margin"] = round(r["margin"] - base["margin"], 4)
            out.append({"arm": "random", "layer": lyr, "alpha": 0.2, **r})
            # Reversed direction; should push toward sycophancy.
            r = run(scaled_steer(-v, lyr, 0.2, ref))
            r["d_margin"] = round(r["margin"] - base["margin"], 4)
            out.append({"arm": "reversed", "layer": lyr, "alpha": 0.2, **r})
            print(f"L{lyr} controls done", flush=True)

    import pandas as pd

    path = Path(OUT_DIR) / "sweep" / f"english__{short}__{source}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet(path)
    RESULTS.commit()
    return out


@APP.function(
    image=IMAGE,
    gpu="A100-40GB",
    volumes={HF_DIR: HF_CACHE, OUT_DIR: RESULTS},
    secrets=[HF_SECRET],
    timeout=60 * 60,
    scaledown_window=60,
)
def dual_stance(
    model_id: str,
    layer: int,
    alphas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8),
    source: str = "bench",
) -> list[dict]:
    """The specificity gate: does the vector resist falsehood, or resist agreement?

    Reports, per alpha:
      resist_wrong       - prefers disagreement when the user is wrong (good)
      resist_right       - prefers disagreement when the user is right (bad)
      resist_surprising  - prefers disagreement with a counterintuitive truth (bad)
      contrarianism      - d(resist_surprising) / d(resist_wrong)
    """
    import sys

    sys.path.insert(0, "/root/src")
    import pandas as pd
    import torch

    from xsyc import dualstance as DS
    from xsyc.scoring import encode_items, scaled_steer, score_batch, steering, token_budget_for

    model, tok = _load_model(model_id)
    budget = token_budget_for(model)
    short = model_id.split("/")[-1]
    blob = torch.load(Path(OUT_DIR) / "vectors" / f"v_{source}__{short}.pt", weights_only=False)
    v, ref = blob[layer]["vector"], blob[layer]["ref_norm"]

    items_meta = DS.build()
    items, meta = [], []
    for it in items_meta:
        items.append((it.prompt, it.agree))
        meta.append((it.id, it.condition, "agree"))
        items.append((it.prompt, it.disagree))
        meta.append((it.id, it.condition, "disagree"))
    enc = encode_items(tok, items)
    batches = list(_batches(items, [len(i) for i, _ in enc], budget))

    frame = pd.DataFrame(meta, columns=["id", "condition", "stance"])

    def run(steer):
        scores = torch.full((len(items),), float("nan"))
        with steering(model, steer):
            for idx in batches:
                scores[idx] = score_batch(model, tok, encoded=[enc[i] for i in idx])
        assert not torch.isnan(scores).any()
        f = frame.copy()
        f["score"] = scores.numpy()
        w = f.pivot_table(index=["id", "condition"], columns="stance", values="score").reset_index()
        w["resists"] = w["disagree"] > w["agree"]
        g = w.groupby("condition")["resists"].mean()
        return float(g["wrong"]), float(g["right"]), float(g["surprising"])

    base_w, base_r, base_s = run(None)
    out = [
        {
            "alpha": 0.0,
            "resist_wrong": round(base_w, 4),
            "resist_right": round(base_r, 4),
            "resist_surprising": round(base_s, 4),
            "d_wrong": 0.0,
            "d_right": 0.0,
            "d_surprising": 0.0,
            "contrarianism": None,
        }
    ]
    for a in alphas:
        if a == 0.0:
            continue
        w, r, s = run(scaled_steer(v, layer, a, ref))
        dw, dr, ds = w - base_w, r - base_r, s - base_s
        out.append(
            {
                "alpha": a,
                "resist_wrong": round(w, 4),
                "resist_right": round(r, 4),
                "resist_surprising": round(s, 4),
                "d_wrong": round(dw, 4),
                "d_right": round(dr, 4),
                "d_surprising": round(ds, 4),
                # Over-correction per unit of real correction, on the arm with
                # headroom (resist_right has none).
                "contrarianism": round(ds / dw, 3) if abs(dw) > 1e-9 else None,
            }
        )
        print(f"a={a}: wrong={w:.3f} right={r:.3f} surprising={s:.3f}", flush=True)

    path = Path(OUT_DIR) / "sweep" / f"dualstance__{short}__{source}__L{layer}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet(path)
    RESULTS.commit()
    return out


@APP.function(
    image=IMAGE,
    gpu="A100-40GB",
    volumes={HF_DIR: HF_CACHE, OUT_DIR: RESULTS},
    secrets=[HF_SECRET],
    timeout=60 * 120,
    scaledown_window=60,
)
def probe_ci(
    model_id: str,
    specs: list[dict],
    langs: list[str] | None = None,
    n_boot: int = 4000,
) -> dict:
    """Paired bootstrap intervals for the specificity probe, English and translated.

    Recomputes the per-item `resists` indicator under base and under each
    steering spec, persists it, and bootstraps. The resampling unit is the
    proposition; base and steered are read from the same resampled
    propositions, so proposition difficulty cancels in the difference.

    `specs` entries are {"source": "bench"|"fact", "layer": int, "alpha": float}.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import numpy as np
    import pandas as pd
    import torch

    from xsyc import dualstance as DS
    from xsyc.scoring import encode_items, scaled_steer, score_batch, steering, token_budget_for

    langs = PROBE_LANGS if langs is None else langs
    model, tok = _load_model(model_id)
    budget = token_budget_for(model)
    short = model_id.split("/")[-1]
    RESULTS.reload()

    vecs = {}
    for sp in specs:
        blob = torch.load(
            Path(OUT_DIR) / "vectors" / f"v_{sp['source']}__{short}.pt", weights_only=False
        )
        vecs[sp["source"]] = blob

    def per_item(items, meta):
        """One (id, condition) row per proposition, with base and steered flags."""
        enc = encode_items(tok, items)
        batches = list(_batches(items, [len(i) for i, _ in enc], budget))
        frame = pd.DataFrame(meta, columns=["id", "condition", "stance"])

        def run(steer):
            sc = torch.full((len(items),), float("nan"))
            with steering(model, steer):
                for idx in batches:
                    sc[idx] = score_batch(model, tok, encoded=[enc[i] for i in idx])
            assert not torch.isnan(sc).any()
            f = frame.copy()
            f["score"] = sc.numpy()
            w = f.pivot_table(index=["id", "condition"], columns="stance",
                              values="score").reset_index()
            return (w["disagree"] > w["agree"]).to_numpy(), w[["id", "condition"]]

        base, key = run(None)
        out = key.copy()
        out["base"] = base
        for sp in specs:
            lyr, a = sp["layer"], sp["alpha"]
            blob = vecs[sp["source"]]
            flags, _ = run(scaled_steer(blob[lyr]["vector"], lyr, a, blob[lyr]["ref_norm"]))
            out[f"{sp['source']}_L{lyr}_a{a}"] = flags
        return out

    # English probe from the authored source; the others from `translate_probe`.
    en_items, en_meta = [], []
    for it in DS.build():
        en_items.append((it.prompt, it.agree))
        en_meta.append((it.id, it.condition, "agree"))
        en_items.append((it.prompt, it.disagree))
        en_meta.append((it.id, it.condition, "disagree"))
    frames = {"en": per_item(en_items, en_meta)}
    print("en done", flush=True)

    for lang in langs:
        df = pd.read_parquet(Path(OUT_DIR) / "probe_i18n" / f"{lang}.parquet")
        items, meta = [], []
        for r in df.itertuples():
            items.append((r.prompt, r.agree))
            meta.append((r.id, r.condition, "agree"))
            items.append((r.prompt, r.disagree))
            meta.append((r.id, r.condition, "disagree"))
        frames[lang] = per_item(items, meta)
        print(f"{lang} done", flush=True)

    rng = np.random.default_rng(0)
    cols = [f"{sp['source']}_L{sp['layer']}_a{sp['alpha']}" for sp in specs]
    out = []
    for lang, f in frames.items():
        f = f.copy()
        f["language"] = lang
        for cond, sub in f.groupby("condition"):
            b = sub["base"].to_numpy(dtype=float)
            n = len(b)
            idx = rng.integers(0, n, size=(n_boot, n))
            row = {"language": lang, "condition": cond, "n": n,
                   "base_rate": round(float(b.mean()), 4)}
            for c in cols:
                s = sub[c].to_numpy(dtype=float)
                d = float(s.mean() - b.mean())
                boot = s[idx].mean(axis=1) - b[idx].mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                row[f"{c}_rate"] = round(float(s.mean()), 4)
                row[f"{c}_d"] = round(d, 4)
                row[f"{c}_lo"] = round(float(lo), 4)
                row[f"{c}_hi"] = round(float(hi), 4)
            out.append(row)

    path = Path(OUT_DIR) / "probe_i18n" / f"probe_ci__{short}.parquet"
    pd.DataFrame(out).to_parquet(path)
    pd.concat([f.assign(language=lg) for lg, f in frames.items()]).to_parquet(
        Path(OUT_DIR) / "probe_i18n" / f"probe_items__{short}.parquet"
    )
    RESULTS.commit()
    return {"rows": out, "n_boot": n_boot}


@APP.function(
    image=IMAGE,
    gpu="A100-40GB",
    volumes={HF_DIR: HF_CACHE, OUT_DIR: RESULTS},
    secrets=[HF_SECRET],
    timeout=60 * 180,
    scaledown_window=60,
)
def targetnorm_sweep(
    langs: list[str],
    model_id: str,
    layer: int = 16,
    alpha: float = 0.2,
    n: int = 300,
    src_tag: str = "oracle",
    tag: str = "targetnorm",
) -> list[dict]:
    """English direction, English layer and alpha, target language's own reference norm.

    `transfer_sweep` scales alpha by the English mean activation norm
    everywhere. This condition changes only the reference norm, so any
    difference from the transfer condition is dose. Reference norms are read
    from the target-language ("oracle") sweep's shards, which computed them
    from the same 396 extract items.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import pandas as pd
    import torch

    from xsyc import data as D
    from xsyc.scoring import encode_items, scaled_steer, score_batch, steering, token_budget_for

    model, tok = _load_model(model_id)
    budget = token_budget_for(model)
    short = model_id.split("/")[-1]
    RESULTS.reload()
    blob = torch.load(Path(OUT_DIR) / "vectors" / f"v_bench__{short}.pt", weights_only=False)
    v_en, ref_en = blob[layer]["vector"], blob[layer]["ref_norm"]

    rows = []
    for lang in langs:
        path = Path(OUT_DIR) / tag / f"{short}__{lang}.parquet"
        if path.exists():
            rows.append(pd.read_parquet(path).iloc[0].to_dict())
            continue

        t0 = time.time()
        src = Path(OUT_DIR) / src_tag / f"{short}__{lang}.parquet"
        ref_l = float(pd.read_parquet(src).iloc[0]["ref_norm"])

        recs = D.load_language(lang, DATA_DIR)
        eva = D.stratified_sample(D.take(recs, "eval"), n)
        items = []
        for r in eva:
            items.append((r.prompt, r.sycophantic))
            items.append((r.prompt, r.not_sycophantic))
        enc = encode_items(tok, items)
        batches = list(_batches(items, [len(i) for i, _ in enc], budget))
        is_syc = torch.arange(len(items)) % 2 == 0

        def margin(steer):
            sc = torch.full((len(items),), float("nan"))
            with steering(model, steer):
                for idx in batches:
                    sc[idx] = score_batch(model, tok, encoded=[enc[i] for i in idx])
            assert not torch.isnan(sc).any()
            return float((sc[~is_syc] - sc[is_syc]).mean())

        m_base = margin(None)
        m_tn = margin(scaled_steer(v_en, layer, alpha, ref_l))

        row = {
            "language": lang,
            "resource_tier": eva[0].resource_tier,
            "ref_norm": round(ref_l, 4),
            "ref_ratio": round(ref_l / ref_en, 4),
            "margin_base": round(m_base, 4),
            "margin_targetnorm": round(m_tn, 4),
            "d_targetnorm": round(m_tn - m_base, 4),
        }
        rows.append(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_parquet(path)
        RESULTS.commit()
        print(f"{lang}: ref_ratio={row['ref_ratio']:.3f} "
              f"d={row['d_targetnorm']:+.4f} ({time.time() - t0:.0f}s)", flush=True)
    return rows


@APP.function(image=IMAGE, volumes={OUT_DIR: RESULTS}, cpu=4.0, memory=8192,
              timeout=1800, scaledown_window=60)
def probe_heldout_ci(model_id: str, n_boot: int = 4000, first_ext: int = 40) -> dict:
    """The probe result restricted to propositions the selection gate never saw.

    `v_fact`'s alpha was chosen with a specificity gate on the first 40
    propositions (`dualstance.FACTS`); `FACTS_EXT` was added afterwards. This
    recomputes the raw falsehood arm and the specificity-adjusted net on ids
    `ds_040` and above only. Reads the persisted per-item frame; no GPU.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import numpy as np
    import pandas as pd

    RESULTS.reload()
    short = model_id.split("/")[-1]
    df = pd.read_parquet(Path(OUT_DIR) / "probe_i18n" / f"probe_items__{short}.parquet")
    cols = [c for c in df.columns if c not in ("id", "condition", "base", "language")]
    df = df[df["condition"].isin(["wrong", "right"])].copy()
    df["prop"] = df["id"].str.extract(r"ds_(\d+)_")[0].astype(int)
    held = df[df["prop"] >= first_ext]

    rng = np.random.default_rng(0)
    out = []
    for lang, sub in held.groupby("language"):
        w = sub[sub["condition"] == "wrong"].sort_values("prop").reset_index(drop=True)
        r = sub[sub["condition"] == "right"].sort_values("prop").reset_index(drop=True)
        assert (w["prop"].to_numpy() == r["prop"].to_numpy()).all(), lang
        n = len(w)
        idx = rng.integers(0, n, size=(n_boot, n))
        wb, rb = w["base"].to_numpy(float), r["base"].to_numpy(float)
        row = {"language": lang, "n": n, "base_rate": round(float(wb.mean()), 4)}
        for c in cols:
            ws, rs = w[c].to_numpy(float), r[c].to_numpy(float)
            raw = float(ws.mean() - wb.mean())
            net = raw - float(rs.mean() - rb.mean())
            braw = ws[idx].mean(axis=1) - wb[idx].mean(axis=1)
            bnet = braw - (rs[idx].mean(axis=1) - rb[idx].mean(axis=1))
            for tag, val, boot in (("raw", raw, braw), ("net", net, bnet)):
                lo, hi = np.percentile(boot, [2.5, 97.5])
                row[f"{c}_{tag}"] = round(val, 4)
                row[f"{c}_{tag}_lo"] = round(float(lo), 4)
                row[f"{c}_{tag}_hi"] = round(float(hi), 4)
        out.append(row)

    pd.DataFrame(out).to_parquet(
        Path(OUT_DIR) / "probe_i18n" / f"probe_heldout_ci__{short}.parquet"
    )
    RESULTS.commit()
    return {"n_boot": n_boot, "excluded_below": first_ext, "rows": out}


@APP.function(image=IMAGE, volumes={OUT_DIR: RESULTS}, cpu=4.0, memory=8192,
              timeout=1800, scaledown_window=60)
def english_heldout(model_id: str, tag: str = "transfer") -> dict:
    """English benchmark numbers on the held-out evaluation sample.

    Reads the English transfer shard and returns the baseline margin and
    sycophancy rate, plus margin, d_resist, relative gain and sycophancy rate
    per steering condition. These, not the development sweep, are the English
    numbers to report.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import pandas as pd

    RESULTS.reload()
    short = model_id.split("/")[-1]
    df = pd.read_parquet(Path(OUT_DIR) / tag / f"{short}__en.parquet")
    w = df.pivot_table(index=["id", "condition"], columns="arm", values="score").reset_index()
    w["margin"] = w["nonsyc"] - w["syc"]
    base = w[w["condition"] == "none"].set_index("id")["margin"]
    out = {"n_items": int(len(base)), "base_margin": round(float(base.mean()), 4),
           "base_syc_rate": round(float((base < 0).mean()), 4)}
    for cond in sorted(set(w["condition"]) - {"none"}):
        m = w[w["condition"] == cond].set_index("id")["margin"].loc[base.index]
        out[cond] = {
            "margin": round(float(m.mean()), 4),
            "d_resist": round(float((m - base).mean()), 4),
            "rel_gain_pct": round(100 * float((m - base).mean()) / float(base.mean()), 1),
            "syc_rate": round(float((m < 0).mean()), 4),
        }
    return out


@APP.function(image=IMAGE, volumes={OUT_DIR: RESULTS}, cpu=4.0, memory=8192,
              timeout=1800, scaledown_window=60)
def probe_net_ci(model_id: str, n_boot: int = 4000) -> list[dict]:
    """Bootstrap the specificity-adjusted probe effect. CPU only.

    Reporting only the falsehood arm lets a generic increase in disagreement
    read as factual resistance. The quantity that cannot be faked that way is

        net = d(corrects a falsehood) - d(disagrees with a true claim)

    which is the change in how far the model separates a false user claim from
    a true one. Proposition `i` appears in both arms as `ds_{i}_wrong` and
    `ds_{i}_right`, so a resample of propositions carries both arms with it and
    the pairing is exact.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import numpy as np
    import pandas as pd

    RESULTS.reload()
    short = model_id.split("/")[-1]
    df = pd.read_parquet(Path(OUT_DIR) / "probe_i18n" / f"probe_items__{short}.parquet")
    cols = [c for c in df.columns if c not in ("id", "condition", "base", "language")]

    df = df[df["condition"].isin(["wrong", "right"])].copy()
    df["prop"] = df["id"].str.extract(r"ds_(\d+)_")[0].astype(int)

    rng = np.random.default_rng(0)
    out = []
    for lang, sub in df.groupby("language"):
        w = sub[sub["condition"] == "wrong"].sort_values("prop").reset_index(drop=True)
        r = sub[sub["condition"] == "right"].sort_values("prop").reset_index(drop=True)
        assert (w["prop"].to_numpy() == r["prop"].to_numpy()).all(), lang
        n = len(w)
        idx = rng.integers(0, n, size=(n_boot, n))
        wb, rb = w["base"].to_numpy(float), r["base"].to_numpy(float)
        row = {"language": lang, "n": n}
        for c in cols:
            ws, rs = w[c].to_numpy(float), r[c].to_numpy(float)
            net = (ws.mean() - wb.mean()) - (rs.mean() - rb.mean())
            boot = ((ws[idx].mean(axis=1) - wb[idx].mean(axis=1))
                    - (rs[idx].mean(axis=1) - rb[idx].mean(axis=1)))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            row[f"{c}_net"] = round(float(net), 4)
            row[f"{c}_net_lo"] = round(float(lo), 4)
            row[f"{c}_net_hi"] = round(float(hi), 4)
        out.append(row)

    pd.DataFrame(out).to_parquet(
        Path(OUT_DIR) / "probe_i18n" / f"probe_net_ci__{short}.parquet"
    )
    RESULTS.commit()
    return out


@APP.function(image=IMAGE, volumes={OUT_DIR: RESULTS}, timeout=600, scaledown_window=60)
def compare_vectors(model_id: str) -> dict:
    """Cosine between v_bench and v_fact at each layer. CPU only.

    A near-parallel pair would make the behavioural dissociation between them
    suspect.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import torch
    import torch.nn.functional as F

    short = model_id.split("/")[-1]
    b = torch.load(Path(OUT_DIR) / "vectors" / f"v_bench__{short}.pt", weights_only=False)
    f = torch.load(Path(OUT_DIR) / "vectors" / f"v_fact__{short}.pt", weights_only=False)
    return {
        str(lyr): round(float(F.cosine_similarity(b[lyr]["vector"], f[lyr]["vector"], dim=0)), 4)
        for lyr in b["layers"]
    }


# --- Phase 2: 38-language transfer ----------------------------------------

# Frozen by the English-only dev sweep, redone per model (Qwen has 28 layers
# to Llama's 32). v_bench: alpha held at 0.2 on both models and only the layer
# chosen by the margin criterion at that alpha; not the grid argmax. v_fact: on
# Llama the best-margin setting passing the probe gate; on Qwen the same
# relative depth with alpha 0.2.
def conditions_for(model_id: str, variant: str = "frozen") -> list[dict]:
    short = model_id.split("/")[-1]
    if variant == "argmax":
        # Appendix check: v_bench at the argmax of dev margin gain over the full
        # layer x alpha grid. Baseline, v_bench and the random control only.
        if short.startswith("Llama-3.1-8B"):
            bench_l, bench_a = 16, 0.8
        elif short.startswith("Qwen2.5-7B"):
            bench_l, bench_a = 21, 0.8
        else:
            raise ValueError(f"no argmax setting for {short}")
        return [
            {"name": "none", "source": None, "layer": None, "alpha": 0.0},
            {"name": "v_bench", "source": "bench", "layer": bench_l, "alpha": bench_a},
            {"name": "random", "source": "bench", "layer": bench_l, "alpha": bench_a,
             "mode": "random"},
        ]
    if short.startswith("Llama-3.1-8B"):
        bench_l, bench_a, fact_l, fact_a = 16, 0.2, 8, 0.4
    elif short.startswith("Qwen2.5-7B"):
        bench_l, bench_a, fact_l, fact_a = 14, 0.2, 7, 0.2
    else:
        raise ValueError(f"no frozen selection for {short}; run sweep_english first")
    return [
        {"name": "none", "source": None, "layer": None, "alpha": 0.0},
        {"name": "v_bench", "source": "bench", "layer": bench_l, "alpha": bench_a},
        {"name": "v_fact", "source": "fact", "layer": fact_l, "alpha": fact_a},
        {"name": "random", "source": "bench", "layer": bench_l, "alpha": bench_a,
         "mode": "random"},
        {"name": "reversed", "source": "bench", "layer": bench_l, "alpha": bench_a,
         "mode": "reversed"},
    ]


CONDITIONS = conditions_for("meta-llama/Llama-3.1-8B-Instruct")


@APP.function(
    image=IMAGE,
    gpu="A100-40GB",
    volumes={HF_DIR: HF_CACHE, OUT_DIR: RESULTS},
    secrets=[HF_SECRET],
    timeout=60 * 120,
    scaledown_window=60,
)
def transfer_sweep(
    langs: list[str],
    model_id: str,
    n: int = 300,
    tag: str = "transfer",
    variant: str = "frozen",
) -> dict:
    """Apply the English-selected vectors, unchanged, to every language.

    Encodings are built once per language and reused across conditions. One
    parquet shard per language, committed before the next language starts.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import pandas as pd
    import torch

    from xsyc import data as D
    from xsyc.scoring import (
        encode_items,
        scaled_steer,
        score_batch,
        steering,
        token_budget_for,
        unit,
    )

    conds = conditions_for(model_id, variant)
    model, tok = _load_model(model_id)
    budget = token_budget_for(model)
    short = model_id.split("/")[-1]

    blobs = {
        src: torch.load(
            Path(OUT_DIR) / "vectors" / f"v_{src}__{short}.pt", weights_only=False
        )
        for src in ("bench", "fact")
    }

    # One random draw reused for every language, so the control is the same
    # direction throughout.
    torch.manual_seed(0)
    rand_dir = unit(torch.randn_like(blobs["bench"][conds[1]["layer"]]["vector"]))

    def steer_for(cond):
        if cond["source"] is None:
            return None
        blob = blobs[cond["source"]]
        lyr = cond["layer"]
        v, ref = blob[lyr]["vector"], blob[lyr]["ref_norm"]
        mode = cond.get("mode")
        if mode == "random":
            v = rand_dir
        elif mode == "reversed":
            v = -v
        return scaled_steer(v, lyr, cond["alpha"], ref)

    done, t_start = [], time.time()
    for lang in langs:
        path = Path(OUT_DIR) / tag / f"{short}__{lang}.parquet"
        if path.exists():
            done.append(lang)
            continue

        records = D.stratified_sample(D.take(D.load_language(lang, DATA_DIR), "eval"), n)
        items, meta = [], []
        for r in records:
            items.append((r.prompt, r.sycophantic))
            meta.append((r.id, r.category, r.sensitivity, "syc"))
            items.append((r.prompt, r.not_sycophantic))
            meta.append((r.id, r.category, r.sensitivity, "nonsyc"))

        enc = encode_items(tok, items)
        batches = list(_batches(items, [len(i) for i, _ in enc], budget))

        frames = []
        t0 = time.time()
        for cond in conds:
            scores = torch.full((len(items),), float("nan"))
            with steering(model, steer_for(cond)):
                for idx in batches:
                    scores[idx] = score_batch(model, tok, encoded=[enc[i] for i in idx])
            assert not torch.isnan(scores).any(), f"{lang}/{cond['name']}: unscored"
            f = pd.DataFrame(meta, columns=["id", "category", "sensitivity", "arm"])
            f["score"] = scores.numpy()
            f["condition"] = cond["name"]
            frames.append(f)

        df = pd.concat(frames)
        df["language"] = lang
        df["resource_tier"] = records[0].resource_tier
        df["model"] = model_id
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        RESULTS.commit()
        done.append(lang)
        print(
            f"{lang}: {len(conds)} conditions x {len(items)} passes "
            f"in {time.time() - t0:.0f}s  ({len(done)}/{len(langs)})",
            flush=True,
        )

    return {
        "languages": len(done),
        "conditions": [c["name"] for c in conds],
        "n_per_lang": n,
        "minutes": round((time.time() - t_start) / 60, 1),
    }


@APP.function(
    image=IMAGE, volumes={OUT_DIR: RESULTS}, timeout=60 * 15, scaledown_window=60
)
def aggregate_transfer(tag: str = "transfer") -> dict:
    """Turn the transfer shards into the paper's primary tables.

    d_resist(l, c) = margin(l, c) - margin(l, none)
    T(l, c)        = d_resist(l, c) / d_resist(English, c)

    T = 1 means the vector works as well in this language as in English, 0
    means no transfer, negative means it backfires.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import pandas as pd

    # A container's view of a Volume is snapshotted at mount time; reload so a
    # run launched while a sweep is still writing does not read a stale view.
    RESULTS.reload()

    paths = sorted((Path(OUT_DIR) / tag).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no shards under {tag}")
    df = pd.concat(pd.read_parquet(p) for p in paths)

    n_cond = df.groupby("language")["condition"].nunique()
    incomplete = n_cond[n_cond != len(CONDITIONS)]
    if len(incomplete):
        raise ValueError(f"languages missing conditions: {incomplete.to_dict()}")

    wide = df.pivot_table(
        index=["language", "resource_tier", "sensitivity", "id", "condition"],
        columns="arm",
        values="score",
    ).reset_index()
    wide["margin"] = wide["nonsyc"] - wide["syc"]
    wide["is_syc"] = wide["syc"] > wide["nonsyc"]

    per = (
        wide.groupby(["resource_tier", "language", "condition"])
        .agg(margin=("margin", "mean"), syc_rate=("is_syc", "mean"), n=("id", "size"))
        .reset_index()
    )
    base = per[per.condition == "none"].set_index("language")["margin"]
    per["d_resist"] = per["margin"] - per["language"].map(base)

    en = per[per.language == "en"].set_index("condition")["d_resist"]
    per["transfer_ratio"] = per.apply(
        lambda r: r["d_resist"] / en[r["condition"]] if en.get(r["condition"], 0) else None,
        axis=1,
    )

    tier = (
        per[per.condition != "none"]
        .groupby(["condition", "resource_tier"])
        .agg(d_resist=("d_resist", "mean"), transfer=("transfer_ratio", "mean"))
        .reset_index()
    )

    sens = (
        wide.groupby(["condition", "sensitivity", "resource_tier"])["margin"]
        .mean()
        .reset_index()
    )
    sens_base = sens[sens.condition == "none"].set_index(["sensitivity", "resource_tier"])["margin"]
    sens["d_resist"] = sens.apply(
        lambda r: r["margin"] - sens_base[(r["sensitivity"], r["resource_tier"])], axis=1
    )

    return {
        "n_languages": int(df["language"].nunique()),
        "conditions": sorted(df["condition"].unique()),
        "per_language": per.to_dict("records"),
        "by_tier": tier.to_dict("records"),
        "by_sensitivity": sens[sens.condition != "none"].to_dict("records"),
    }


# --- Phase 4: target-language ("oracle") vectors and correlates -----------


@APP.function(
    image=IMAGE,
    gpu="A100-40GB",
    volumes={HF_DIR: HF_CACHE, OUT_DIR: RESULTS},
    secrets=[HF_SECRET],
    timeout=60 * 150,
    scaledown_window=60,
)
def oracle_sweep(
    langs: list[str],
    model_id: str,
    layer: int = 16,
    alpha: float = 0.2,
    n: int = 300,
    tag: str = "oracle",
) -> dict:
    """Per-language oracle vector, its effect, and the transfer correlates.

    "Oracle" here means the target-language vector of the paper: a vector
    extracted from that language's own `extract` split. The split is parallel
    across all 38 languages, so each oracle vector comes from the same 396
    items as the English vector, in the benchmark authors' translations.

    The oracle is the ceiling. If it also fails, the direction is not linearly
    available in that language; if it works, the direction exists and the
    English vector does not point at it.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import pandas as pd
    import torch
    import torch.nn.functional as F

    from xsyc import data as D
    from xsyc.scoring import (
        collect_activations,
        difference_vector,
        encode_items,
        scaled_steer,
        score_batch,
        steering,
        token_budget_for,
    )

    model, tok = _load_model(model_id)
    budget = token_budget_for(model)
    short = model_id.split("/")[-1]
    en_blob = torch.load(
        Path(OUT_DIR) / "vectors" / f"v_bench__{short}.pt", weights_only=False
    )
    v_en = en_blob[layer]["vector"]

    rows, vectors = [], {}
    for lang in langs:
        path = Path(OUT_DIR) / tag / f"{short}__{lang}.parquet"
        if path.exists():
            rows.append(pd.read_parquet(path).iloc[0].to_dict())
            continue

        t0 = time.time()
        recs = D.load_language(lang, DATA_DIR)

        # --- oracle vector from this language's own extract split
        ext = D.take(recs, "extract")
        a_syc = collect_activations(
            model, tok, [(r.prompt, r.sycophantic) for r in ext], [layer]
        )[layer]
        a_non = collect_activations(
            model, tok, [(r.prompt, r.not_sycophantic) for r in ext], [layer]
        )[layer]
        v_l = difference_vector(a_non, a_syc)
        ref_l = float(torch.cat([a_syc, a_non]).norm(dim=-1).mean())
        vectors[lang] = v_l

        # --- fertility: tokens per item on the same parallel IDs
        fert = float(
            sum(len(tok.encode(r.prompt, add_special_tokens=False)) for r in ext)
        ) / len(ext)

        # --- effect of the oracle vector on this language's eval items
        eva = D.stratified_sample(D.take(recs, "eval"), n)
        items = []
        for r in eva:
            items.append((r.prompt, r.sycophantic))
            items.append((r.prompt, r.not_sycophantic))
        enc = encode_items(tok, items)
        batches = list(_batches(items, [len(i) for i, _ in enc], budget))
        is_syc = torch.arange(len(items)) % 2 == 0

        def margin(steer):
            sc = torch.full((len(items),), float("nan"))
            with steering(model, steer):
                for idx in batches:
                    sc[idx] = score_batch(model, tok, encoded=[enc[i] for i in idx])
            assert not torch.isnan(sc).any()
            return float((sc[~is_syc] - sc[is_syc]).mean())

        m_base = margin(None)
        m_oracle = margin(scaled_steer(v_l, layer, alpha, ref_l))

        row = {
            "language": lang,
            "resource_tier": eva[0].resource_tier,
            "cos_en": round(float(F.cosine_similarity(v_en, v_l, dim=0)), 4),
            "fertility": round(fert, 2),
            "v_norm": round(float(v_l.norm()), 4),
            "ref_norm": round(ref_l, 4),
            "rel_norm": round(float(v_l.norm()) / ref_l, 4),
            "margin_base": round(m_base, 4),
            "margin_oracle": round(m_oracle, 4),
            "d_oracle": round(m_oracle - m_base, 4),
        }
        rows.append(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_parquet(path)
        RESULTS.commit()
        print(f"{lang}: cos={row['cos_en']:.3f} fert={fert:.1f} "
              f"d_oracle={row['d_oracle']:+.4f} ({time.time() - t0:.0f}s)", flush=True)

    if vectors:
        torch.save(vectors, Path(OUT_DIR) / "vectors" / f"oracle__{short}__L{layer}.pt")
        RESULTS.commit()
    return {"languages": len(rows), "layer": layer, "alpha": alpha}


@APP.function(
    image=IMAGE, volumes={OUT_DIR: RESULTS}, timeout=60 * 15, scaledown_window=60
)
def correlates(model_id: str, tag: str = "oracle") -> dict:
    """Join oracle results to transfer ratios and report correlations.

    Descriptive only: the 38 languages are not independent (shared families,
    scripts and translation pipeline), so these are correlations, not effects.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import pandas as pd

    RESULTS.reload()
    short = model_id.split("/")[-1]

    orc = pd.concat(
        pd.read_parquet(p) for p in sorted((Path(OUT_DIR) / tag).glob("*.parquet"))
    )
    tdir = "transfer" if tag == "oracle" else "transfer_qwen"
    tr = pd.concat(
        pd.read_parquet(p) for p in sorted((Path(OUT_DIR) / tdir).glob("*.parquet"))
    )
    w = tr.pivot_table(
        index=["language", "id", "condition"], columns="arm", values="score"
    ).reset_index()
    w["margin"] = w["nonsyc"] - w["syc"]
    per = w.groupby(["language", "condition"])["margin"].mean().unstack()
    per["d_bench"] = per["v_bench"] - per["none"]
    en_d = float(per.loc["en", "d_bench"])
    per["transfer"] = per["d_bench"] / en_d

    df = orc.merge(
        per[["d_bench", "transfer"]].reset_index(), on="language", how="inner"
    )
    df["oracle_gap"] = df["d_oracle"] - df["d_bench"]

    num = ["cos_en", "fertility", "rel_norm", "margin_base", "d_oracle"]

    # English is excluded: by construction transfer = 1 and cos_en = 1, and
    # leaving it in anchors every fit.
    ind = df[df.language != "en"]
    corr = {c: round(float(ind["transfer"].corr(ind[c])), 3) for c in num}
    corr_spearman = {
        c: round(float(ind["transfer"].corr(ind[c], method="spearman")), 3) for c in num
    }
    corr_with_en = {c: round(float(df["transfer"].corr(df[c])), 3) for c in num}

    df["oracle_wins"] = df["d_oracle"] > df["d_bench"]

    by_tier = df.groupby("resource_tier")[
        ["transfer", "d_bench", "d_oracle", "cos_en", "fertility"]
    ].mean()

    return {
        "n": len(df),
        "n_excl_en": len(ind),
        "pearson_vs_transfer": corr,
        "spearman_vs_transfer": corr_spearman,
        "pearson_including_en": corr_with_en,
        "oracle_wins": int(df["oracle_wins"].sum()),
        "oracle_ratio_mean": round(float((df["d_oracle"] / df["d_bench"]).replace(
            [float("inf"), float("-inf")], float("nan")).dropna().mean()), 3),
        "by_tier": by_tier.round(4).to_dict("index"),
        "per_language": df.round(4).to_dict("records"),
    }


@APP.function(image=IMAGE, volumes={OUT_DIR: RESULTS}, timeout=600, scaledown_window=60)
def backcheck() -> dict:
    """Compare back-translations to originals, focusing on numbers.

    MT drops or garbles digits quietly. Scores the fraction of numeric tokens
    in the original that survive the round trip and returns examples to read.
    """
    import re

    import pandas as pd

    RESULTS.reload()
    out, examples = {}, {}
    # The forward translation is returned too: a number missing from the
    # back-translation may be a back-translation artifact.
    fwd = {}
    for p in sorted((Path(OUT_DIR) / "probe_i18n").glob("*.parquet")):
        if "backcheck" in p.name or "results__" in p.name:
            continue
        d = pd.read_parquet(p)
        fwd[p.stem] = d[d.condition == "wrong"].head(2)[["id", "disagree", "agree"]].to_dict("records")
    for p in sorted((Path(OUT_DIR) / "probe_i18n").glob("*__backcheck.parquet")):
        lang = p.name.split("__")[0]
        df = pd.read_parquet(p)
        hits = tot = 0
        for r in df.itertuples():
            nums = re.findall(r"\d[\d,.]*", r.original)
            for nmb in nums:
                tot += 1
                hits += int(nmb in r.back_translated)
        out[lang] = hits / tot if tot else float("nan")
        examples[lang] = [
            (r.original, r.back_translated) for r in df.head(3).itertuples()
        ]
    return {"numeric_match": out, "examples": examples, "forward": fwd}


# --- Cross-lingual specificity probe ---------------------------------------

# Eight languages spanning all three tiers and five scripts, fixed before any
# probe result was seen.
PROBE_LANGS = ["es", "de", "zh", "ar", "hi", "bn", "te", "my"]

NLLB = "facebook/nllb-200-distilled-600M"
NLLB_CODE = {
    "es": "spa_Latn", "de": "deu_Latn", "zh": "zho_Hans", "ar": "arb_Arab",
    "hi": "hin_Deva", "bn": "ben_Beng", "te": "tel_Telu", "my": "mya_Mymr",
    "en": "eng_Latn",
}


@APP.function(image=IMAGE, gpu="A100-40GB", volumes={HF_DIR: HF_CACHE, OUT_DIR: RESULTS},
              secrets=[HF_SECRET], timeout=60*60, scaledown_window=60)
def backcheck_numeric(langs: list[str] = PROBE_LANGS, n: int = 40) -> dict:
    """Numeric-retention check on the `wrong` arm.

    Restricted to propositions containing digits, where a dropped or altered
    number would invert an item's label. The `surprising` arm, which
    `backcheck` samples, has no digits.
    """
    import re
    import sys

    sys.path.insert(0, "/root/src")
    import pandas as pd
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from xsyc import dualstance as DS

    RESULTS.reload()
    orig = {i.id: i.disagree for i in DS.build()}
    tok = AutoTokenizer.from_pretrained(NLLB)
    mt = AutoModelForSeq2SeqLM.from_pretrained(NLLB, torch_dtype=torch.float32).cuda().eval()

    @torch.no_grad()
    def tr(texts, src, tgt, bs=32):
        tok.src_lang = src
        out = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to("cuda")
            g = mt.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                            max_new_tokens=200, num_beams=4)
            out += tok.batch_decode(g, skip_special_tokens=True)
        return out

    res = {}
    for lang in langs:
        df = pd.read_parquet(Path(OUT_DIR) / "probe_i18n" / f"{lang}.parquet")
        cand = [r for r in df[df.condition == "wrong"].itertuples()
                if re.search(r"\d", orig[r.id])][:n]
        back = tr([r.disagree for r in cand], NLLB_CODE[lang], NLLB_CODE["en"])
        hits = tot = 0
        misses = []
        for r, b in zip(cand, back):
            nums = re.findall(r"\d[\d,.]*", orig[r.id])
            ok = True
            for nmb in nums:
                tot += 1
                keep = nmb in b or nmb.replace(",", ".") in b or nmb.replace(",", "") in b
                hits += int(keep)
                ok &= keep
            if not ok:
                misses.append((orig[r.id][:80], b[:80]))
        res[lang] = {"numeric_retention": round(hits / tot, 3) if tot else None,
                     "n_items": len(cand), "n_numbers": tot,
                     "example_misses": misses[:3]}
        print(f"{lang}: {res[lang]['numeric_retention']:.2f} over {tot} numbers", flush=True)
    return res




@APP.function(
    image=IMAGE,
    gpu="A100-40GB",
    volumes={HF_DIR: HF_CACHE, OUT_DIR: RESULTS},
    secrets=[HF_SECRET],
    timeout=60 * 120,
    scaledown_window=60,
)
def translate_probe(langs: list[str] = PROBE_LANGS, back_check: int = 30) -> dict:
    """Translate the dual-stance probe with NLLB, and back-translate a sample.

    NLLB was among the systems used to translate the benchmark itself. Machine
    translation fails quietly on numbers, units and named entities, which is
    what the probe is made of, so a sample of the disagree arm is
    back-translated to English and stored for inspection.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import pandas as pd
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from xsyc import dualstance as DS

    tok = AutoTokenizer.from_pretrained(NLLB)
    mt = AutoModelForSeq2SeqLM.from_pretrained(NLLB, torch_dtype=torch.float32).cuda().eval()

    items = DS.build()
    fields = [("prompt", i.prompt) for i in items]
    fields += [("agree", i.agree) for i in items]
    fields += [("disagree", i.disagree) for i in items]

    @torch.no_grad()
    def batch_translate(texts: list[str], src: str, tgt: str, bs: int = 48) -> list[str]:
        tok.src_lang = src
        out = []
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to("cuda")
            gen = mt.generate(
                **enc,
                forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                max_new_tokens=200,
                num_beams=4,
            )
            out += tok.batch_decode(gen, skip_special_tokens=True)
        return out

    report = {}
    for lang in langs:
        path = Path(OUT_DIR) / "probe_i18n" / f"{lang}.parquet"
        if path.exists():
            report[lang] = "cached"
            continue
        t0 = time.time()
        texts = [v for _, v in fields]
        translated = batch_translate(texts, NLLB_CODE["en"], NLLB_CODE[lang])

        n = len(items)
        df = pd.DataFrame(
            {
                "id": [i.id for i in items],
                "condition": [i.condition for i in items],
                "prompt": translated[:n],
                "agree": translated[n : 2 * n],
                "disagree": translated[2 * n :],
                "language": lang,
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

        # Back-translate a fixed sample of the disagree arm, where the factual
        # content sits, and store it next to the original.
        sample = df.head(back_check)
        back = batch_translate(
            list(sample["disagree"]), NLLB_CODE[lang], NLLB_CODE["en"]
        )
        orig = {i.id: i.disagree for i in items}
        pd.DataFrame(
            {
                "id": sample["id"],
                "original": [orig[i] for i in sample["id"]],
                "back_translated": back,
            }
        ).to_parquet(Path(OUT_DIR) / "probe_i18n" / f"{lang}__backcheck.parquet")
        RESULTS.commit()
        report[lang] = f"{len(df)} items in {time.time() - t0:.0f}s"
        print(f"{lang}: {report[lang]}", flush=True)

    return report


@APP.function(
    image=IMAGE,
    gpu="A100-40GB",
    volumes={HF_DIR: HF_CACHE, OUT_DIR: RESULTS},
    secrets=[HF_SECRET],
    timeout=60 * 120,
    scaledown_window=60,
)
def dual_stance_multilingual(
    model_id: str,
    layer: int,
    alpha: float,
    source: str = "bench",
    langs: list[str] = PROBE_LANGS,
) -> list[dict]:
    """The dual-stance probe on the translated probe files, one setting per call.

    Returns per-language base and steered resist rates for each arm.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import pandas as pd
    import torch

    from xsyc.scoring import encode_items, scaled_steer, score_batch, steering, token_budget_for

    model, tok = _load_model(model_id)
    budget = token_budget_for(model)
    short = model_id.split("/")[-1]
    blob = torch.load(Path(OUT_DIR) / "vectors" / f"v_{source}__{short}.pt", weights_only=False)
    v, ref = blob[layer]["vector"], blob[layer]["ref_norm"]
    RESULTS.reload()

    out = []
    for lang in langs:
        df = pd.read_parquet(Path(OUT_DIR) / "probe_i18n" / f"{lang}.parquet")
        items, meta = [], []
        for r in df.itertuples():
            items.append((r.prompt, r.agree))
            meta.append((r.id, r.condition, "agree"))
            items.append((r.prompt, r.disagree))
            meta.append((r.id, r.condition, "disagree"))
        enc = encode_items(tok, items)
        batches = list(_batches(items, [len(i) for i, _ in enc], budget))
        frame = pd.DataFrame(meta, columns=["id", "condition", "stance"])

        def run(steer):
            sc = torch.full((len(items),), float("nan"))
            with steering(model, steer):
                for idx in batches:
                    sc[idx] = score_batch(model, tok, encoded=[enc[i] for i in idx])
            assert not torch.isnan(sc).any()
            f = frame.copy()
            f["score"] = sc.numpy()
            w = f.pivot_table(index=["id", "condition"], columns="stance",
                              values="score").reset_index()
            w["resists"] = w["disagree"] > w["agree"]
            return w.groupby("condition")["resists"].mean().to_dict()

        base, steered = run(None), run(scaled_steer(v, layer, alpha, ref))
        row = {"language": lang, "source": source, "alpha": alpha}
        for c in ("wrong", "right", "surprising"):
            row[f"base_{c}"] = round(float(base.get(c, float("nan"))), 4)
            row[f"steered_{c}"] = round(float(steered.get(c, float("nan"))), 4)
            row[f"d_{c}"] = round(row[f"steered_{c}"] - row[f"base_{c}"], 4)
        out.append(row)
        print(f"{lang}: d_wrong={row['d_wrong']:+.3f}", flush=True)

    pd.DataFrame(out).to_parquet(
        Path(OUT_DIR) / "probe_i18n" / f"results__{short}__{source}.parquet"
    )
    RESULTS.commit()
    return out


@APP.function(image=IMAGE, volumes={OUT_DIR: RESULTS}, cpu=4.0, memory=8192,
              timeout=60 * 20, scaledown_window=60)
def bootstrap_ci(tag: str = "transfer", n_boot: int = 4000, seed: int = 0) -> dict:
    """Paired bootstrap CIs on per-language d_resist.

    Resamples items with replacement within a language and recomputes the
    steered-minus-baseline difference on the same resampled ids; an unpaired
    bootstrap would re-introduce between-item variance the difference cancels.
    Also reports whether v_bench beats the random control on the same
    resamples, which is the comparison that matters, not whether it beats zero.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import numpy as np
    import pandas as pd

    RESULTS.reload()
    paths = sorted((Path(OUT_DIR) / tag).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(tag)
    df = pd.concat(pd.read_parquet(p) for p in paths)

    w = df.pivot_table(
        index=["language", "resource_tier", "id", "condition"], columns="arm", values="score"
    ).reset_index()
    w["margin"] = w["nonsyc"] - w["syc"]
    wide = w.pivot_table(
        index=["language", "resource_tier", "id"], columns="condition", values="margin"
    ).reset_index()

    print(f"loaded {len(df)} rows from {len(paths)} shards", flush=True)
    rng = np.random.default_rng(seed)
    out = []
    for (lang, tier), g in wide.groupby(["language", "resource_tier"]):
        n = len(g)
        idx = rng.integers(0, n, size=(n_boot, n))
        row = {"language": lang, "resource_tier": tier, "n_items": int(n)}
        for cond in ("v_bench", "v_fact", "random", "reversed"):
            if cond not in g:
                continue
            d = (g[cond] - g["none"]).to_numpy()
            boots = d[idx].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            row[f"{cond}_d"] = round(float(d.mean()), 5)
            row[f"{cond}_lo"] = round(float(lo), 5)
            row[f"{cond}_hi"] = round(float(hi), 5)
        # Difference of differences against the random control, same resamples.
        if "v_bench" in g and "random" in g:
            dd = ((g["v_bench"] - g["none"]) - (g["random"] - g["none"])).to_numpy()
            b = dd[idx].mean(axis=1)
            lo, hi = np.percentile(b, [2.5, 97.5])
            row["vs_random_lo"], row["vs_random_hi"] = round(float(lo), 5), round(float(hi), 5)
            row["beats_random"] = bool(lo > 0)
        out.append(row)
    return {"n_boot": n_boot, "rows": out}


RANDOM_BANK_LANGS = ["en", "es", "de", "he", "bn", "te", "ml", "as", "my", "lo"]


@APP.function(
    image=IMAGE,
    gpu="A100-40GB",
    volumes={HF_DIR: HF_CACHE, OUT_DIR: RESULTS},
    secrets=[HF_SECRET],
    timeout=60 * 300,
    scaledown_window=60,
)
def random_bank(
    model_id: str,
    langs: list[str] | None = None,
    layer: int = 16,
    alpha: float = 0.2,
    k: int = 20,
    n: int = 300,
    tag: str = "randbank",
) -> dict:
    """A bank of matched-norm random directions, so the null has a distribution.

    The published control is a single seed-0 draw, which controls for
    perturbation magnitude but not for where the real effect sits among random
    directions. `k` directions at seeds 100..100+k-1, plus the seed-0 direction
    under its published name. Ten languages spanning all three tiers, including
    the marginal cases. With k=20 the smallest attainable permutation p-value
    is 1/21 = 0.048.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import pandas as pd
    import torch

    from xsyc import data as D
    from xsyc.scoring import (
        encode_items,
        scaled_steer,
        score_batch,
        steering,
        token_budget_for,
        unit,
    )

    langs = RANDOM_BANK_LANGS if langs is None else langs
    model, tok = _load_model(model_id)
    budget = token_budget_for(model)
    short = model_id.split("/")[-1]
    RESULTS.reload()
    blob = torch.load(Path(OUT_DIR) / "vectors" / f"v_bench__{short}.pt", weights_only=False)
    v_bench, ref = blob[layer]["vector"], blob[layer]["ref_norm"]

    # The published draw, made the same way as in transfer_sweep.
    torch.manual_seed(0)
    published = unit(torch.randn_like(v_bench))

    bank = []
    for i in range(k):
        g = torch.Generator(device="cpu").manual_seed(100 + i)
        r = torch.randn(v_bench.shape, generator=g).to(device=v_bench.device, dtype=v_bench.dtype)
        bank.append(unit(r))

    conds = [("none", None), ("v_bench", v_bench), ("random", published)]
    conds += [(f"rand_{i:02d}", d) for i, d in enumerate(bank)]

    done = []
    for lang in langs:
        path = Path(OUT_DIR) / tag / f"{short}__{lang}.parquet"
        if path.exists():
            done.append(lang)
            continue

        records = D.stratified_sample(D.take(D.load_language(lang, DATA_DIR), "eval"), n)
        items, meta = [], []
        for r in records:
            items.append((r.prompt, r.sycophantic))
            meta.append((r.id, r.sensitivity, "syc"))
            items.append((r.prompt, r.not_sycophantic))
            meta.append((r.id, r.sensitivity, "nonsyc"))
        enc = encode_items(tok, items)
        batches = list(_batches(items, [len(i) for i, _ in enc], budget))

        frames, t0 = [], time.time()
        for name, direction in conds:
            steer = None if direction is None else scaled_steer(direction, layer, alpha, ref)
            scores = torch.full((len(items),), float("nan"))
            with steering(model, steer):
                for idx in batches:
                    scores[idx] = score_batch(model, tok, encoded=[enc[i] for i in idx])
            assert not torch.isnan(scores).any(), f"{lang}/{name}: unscored"
            f = pd.DataFrame(meta, columns=["id", "sensitivity", "arm"])
            f["score"] = scores.numpy()
            f["condition"] = name
            frames.append(f)

        df = pd.concat(frames)
        df["language"] = lang
        df["resource_tier"] = records[0].resource_tier
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        RESULTS.commit()
        done.append(lang)
        print(f"{lang}: {len(conds)} conditions in {time.time() - t0:.0f}s "
              f"({len(done)}/{len(langs)})", flush=True)

    return {"languages": done, "k": k, "conditions": len(conds)}


@APP.function(image=IMAGE, volumes={OUT_DIR: RESULTS}, cpu=4.0, memory=8192,
              timeout=1800, scaledown_window=60)
def bank_stats(model_id: str, tag: str = "randbank") -> dict:
    """Where the real effect sits in the random-direction population.

    Rank-based p-value, p = (1 + #{random >= v_bench}) / (k + 1), then a Holm
    step-down across the languages tested (the claim is per-language, not a
    discovery rate).
    """
    import sys

    sys.path.insert(0, "/root/src")
    import numpy as np
    import pandas as pd

    RESULTS.reload()
    short = model_id.split("/")[-1]
    paths = sorted((Path(OUT_DIR) / tag).glob(f"{short}__*.parquet"))
    if not paths:
        raise FileNotFoundError(tag)
    df = pd.concat(pd.read_parquet(p) for p in paths)

    w = df.pivot_table(index=["language", "resource_tier", "id", "condition"],
                       columns="arm", values="score").reset_index()
    w["margin"] = w["nonsyc"] - w["syc"]
    wide = w.pivot_table(index=["language", "resource_tier", "id"],
                         columns="condition", values="margin").reset_index()

    rand_cols = sorted(c for c in wide.columns if str(c).startswith("rand_"))
    out = []
    for (lang, tier), g in wide.groupby(["language", "resource_tier"]):
        base = g["none"]
        d_bench = float((g["v_bench"] - base).mean())
        d_rand = np.array([float((g[c] - base).mean()) for c in rand_cols])
        d_pub = float((g["random"] - base).mean())
        n_ge = int((d_rand >= d_bench).sum())
        out.append({
            "language": lang, "resource_tier": tier, "k": len(rand_cols),
            "d_bench": round(d_bench, 5),
            "d_published_random": round(d_pub, 5),
            "rand_mean": round(float(d_rand.mean()), 5),
            "rand_sd": round(float(d_rand.std(ddof=1)), 5),
            "rand_min": round(float(d_rand.min()), 5),
            "rand_max": round(float(d_rand.max()), 5),
            "published_pctile": round(float((d_rand <= d_pub).mean()), 3),
            "n_rand_ge_bench": n_ge,
            "p_perm": round((1 + n_ge) / (len(rand_cols) + 1), 4),
        })

    # Holm step-down over the languages tested.
    order = sorted(range(len(out)), key=lambda i: out[i]["p_perm"])
    m, prev = len(out), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * out[i]["p_perm"])
        adj = max(adj, prev)
        prev = adj
        out[i]["p_holm"] = round(adj, 4)
        out[i]["holm_sig"] = bool(adj < 0.05)

    pd.DataFrame(out).to_parquet(Path(OUT_DIR) / f"bank_stats__{short}.parquet")
    RESULTS.commit()
    return {"rows": out}


@APP.function(image=IMAGE, volumes={OUT_DIR: RESULTS}, cpu=4.0, memory=8192,
              timeout=1800, scaledown_window=60)
def ratio_ci(tag: str = "transfer", n_boot: int = 4000, seed: int = 0) -> dict:
    """Transfer-ratio intervals that resample the denominator too.

    `bootstrap_ci` intervals cover d_resist within a language; dividing them
    by English's point estimate ignores the denominator's uncertainty. Every
    language is scored on the same 300 item ids, so one resample of ids is
    evaluated for both d_l and d_English, and the shared item-difficulty
    component cancels in the ratio.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import numpy as np
    import pandas as pd

    RESULTS.reload()
    paths = sorted((Path(OUT_DIR) / tag).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(tag)
    df = pd.concat(pd.read_parquet(p) for p in paths)
    w = df.pivot_table(
        index=["language", "resource_tier", "id", "condition"], columns="arm", values="score"
    ).reset_index()
    w["margin"] = w["nonsyc"] - w["syc"]
    wide = w.pivot_table(
        index=["language", "resource_tier", "id"], columns="condition", values="margin"
    ).reset_index()

    # Intersect ids across languages so the resample indexes the same items
    # everywhere.
    ids = None
    for lang, g in wide.groupby("language"):
        s = set(g["id"])
        ids = s if ids is None else (ids & s)
    ids = sorted(ids)
    print(f"{len(ids)} shared ids across {wide['language'].nunique()} languages", flush=True)

    per = {}
    for (lang, tier), g in wide.groupby(["language", "resource_tier"]):
        g = g.set_index("id").loc[ids]
        per[lang] = (tier, (g["v_bench"] - g["none"]).to_numpy())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    en_boot = per["en"][1][idx].mean(axis=1)
    en_point = float(per["en"][1].mean())

    out = []
    for lang, (tier, d) in sorted(per.items()):
        boots = d[idx].mean(axis=1) / en_boot
        lo, hi = np.percentile(boots, [2.5, 97.5])
        naive = np.percentile(d[idx].mean(axis=1), [2.5, 97.5]) / en_point
        out.append({
            "language": lang, "resource_tier": tier,
            "T": round(float(d.mean() / en_point), 4),
            "joint_lo": round(float(lo), 4), "joint_hi": round(float(hi), 4),
            "naive_lo": round(float(naive[0]), 4), "naive_hi": round(float(naive[1]), 4),
            "joint_width": round(float(hi - lo), 4),
            "naive_width": round(float(naive[1] - naive[0]), 4),
        })
    pd.DataFrame(out).to_parquet(Path(OUT_DIR) / f"ratio_ci__{tag}.parquet")
    RESULTS.commit()
    return {"n_boot": n_boot, "n_ids": len(ids), "rows": out}


# --- Aggregation ----------------------------------------------------------


@APP.function(
    image=IMAGE,
    volumes={OUT_DIR: RESULTS},
    timeout=60 * 10,
    scaledown_window=60,
)
def aggregate(tag: str = "baseline") -> dict:
    """Collapse per-language baseline shards into per-language and per-tier tables.

    CPU only. Runs on Modal rather than locally so reported numbers keep a
    single provenance chain.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import pandas as pd

    paths = sorted((Path(OUT_DIR) / tag).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no shards under {tag}")
    df = pd.concat(pd.read_parquet(p) for p in paths)

    wide = df.pivot_table(
        index=["model", "language", "resource_tier", "sensitivity", "id"],
        columns="arm",
        values="score",
    ).reset_index()
    wide["is_syc"] = wide["syc"] > wide["nonsyc"]
    wide["margin"] = wide["nonsyc"] - wide["syc"]

    by_lang = (
        wide.groupby(["resource_tier", "language"])
        .agg(syc_rate=("is_syc", "mean"), margin=("margin", "mean"), n=("id", "size"))
        .reset_index()
    )
    by_tier = wide.groupby("resource_tier")["is_syc"].mean()
    by_sens = wide.pivot_table(
        index="resource_tier", columns="sensitivity", values="is_syc", aggfunc="mean"
    )

    return {
        "n_languages": int(df["language"].nunique()),
        "n_rows": int(len(wide)),
        "by_language": by_lang.to_dict("records"),
        "by_tier": by_tier.to_dict(),
        "by_tier_sensitivity": by_sens.to_dict(),
    }


# --- Scoring --------------------------------------------------------------


def _load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    return model, tok


def _batches(items, tok_estimate, budget: int):
    """Group items into batches under a padded-token budget, sorted by length
    so padding waste stays low."""
    order = sorted(range(len(items)), key=lambda i: tok_estimate[i])
    batch, width = [], 0
    for i in order:
        w = max(width, tok_estimate[i])
        if batch and w * (len(batch) + 1) > budget:
            yield batch
            batch, width = [i], tok_estimate[i]
        else:
            batch.append(i)
            width = w
    if batch:
        yield batch


@APP.function(
    image=IMAGE,
    gpu="A100-40GB",
    volumes={HF_DIR: HF_CACHE, OUT_DIR: RESULTS},
    secrets=[HF_SECRET],
    timeout=60 * 90,
    scaledown_window=60,
)
def score_many(
    langs: list[str],
    model_id: str,
    split: str = "eval",
    n: int | None = 300,
    tag: str = "baseline",
) -> list[dict]:
    """Score several languages in one container, loading the model once.

    One container per language would pay a ~40s weight load each time, which
    at 38 languages exceeds the compute. Each language commits its own shard
    before the next starts.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import pandas as pd
    import torch

    from xsyc import data as D
    from xsyc.scoring import encode_items, score_batch, token_budget_for

    model, tok = _load_model(model_id)
    budget = token_budget_for(model)
    short = model_id.split("/")[-1]
    out: list[dict] = []

    for lang in langs:
        path = Path(OUT_DIR) / tag / f"{short}__{lang}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
        else:
            records = D.take(D.load_language(lang, DATA_DIR), split)
            if n:
                records = D.stratified_sample(records, n)

            items, meta = [], []
            for r in records:
                items.append((r.prompt, r.sycophantic))
                meta.append((r.id, r.category, r.sensitivity, "syc"))
                items.append((r.prompt, r.not_sycophantic))
                meta.append((r.id, r.category, r.sensitivity, "nonsyc"))

            enc = encode_items(tok, items)
            true_len = [len(ids) for ids, _ in enc]
            scores = torch.full((len(items),), float("nan"))
            t0 = time.time()
            for idx in _batches(items, true_len, budget):
                scores[idx] = score_batch(
                    model, tok, encoded=[enc[i] for i in idx]
                )
            assert not torch.isnan(scores).any(), f"{lang}: unscored items"

            df = pd.DataFrame(meta, columns=["id", "category", "sensitivity", "arm"])
            df["score"] = scores.numpy()
            df["language"] = lang
            df["resource_tier"] = records[0].resource_tier
            df["model"] = model_id
            df["tag"] = tag
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            RESULTS.commit()
            print(f"{lang}: {len(items)} passes in {time.time() - t0:.0f}s", flush=True)

        wide = df.pivot_table(index="id", columns="arm", values="score")
        out.append(
            {
                "lang": lang,
                "tier": df["resource_tier"].iloc[0],
                "n": len(wide),
                "syc_rate": round(float((wide["syc"] > wide["nonsyc"]).mean()), 4),
                "margin": round(float((wide["nonsyc"] - wide["syc"]).mean()), 4),
            }
        )
    return out


@APP.function(
    image=IMAGE,
    gpu="A100-40GB",
    volumes={HF_DIR: HF_CACHE, OUT_DIR: RESULTS},
    secrets=[HF_SECRET],
    timeout=60 * 90,
    scaledown_window=60,
)
def score_language(
    lang: str,
    model_id: str,
    split: str = "eval",
    n: int | None = None,
    tag: str = "baseline",
    token_budget: int | None = None,
) -> dict:
    """Baseline forced-choice scoring for one language. Writes a parquet shard."""
    import sys

    sys.path.insert(0, "/root/src")
    import pandas as pd
    import torch

    from xsyc import data as D
    from xsyc.scoring import encode_items, score_batch, token_budget_for

    out_path = Path(OUT_DIR) / tag / f"{model_id.split('/')[-1]}__{lang}.parquet"
    if out_path.exists():
        return {"lang": lang, "status": "cached", "path": str(out_path)}

    records = D.load_language(lang, DATA_DIR)
    records = D.take(records, split)
    if n:
        records = D.stratified_sample(records, n)

    model, tok = _load_model(model_id)
    if token_budget is None:
        token_budget = token_budget_for(model)

    # Two rows per record: sycophantic and non-sycophantic completion.
    items, meta = [], []
    for r in records:
        items.append((r.prompt, r.sycophantic))
        meta.append((r.id, r.category, r.sensitivity, "syc"))
        items.append((r.prompt, r.not_sycophantic))
        meta.append((r.id, r.category, r.sensitivity, "nonsyc"))

    enc = encode_items(tok, items)
    true_len = [len(ids) for ids, _ in enc]
    scores = torch.full((len(items),), float("nan"))

    t0 = time.time()
    for idx in _batches(items, true_len, token_budget):
        scores[idx] = score_batch(model, tok, encoded=[enc[i] for i in idx])
    assert not torch.isnan(scores).any(), "some items were never scored"

    df = pd.DataFrame(meta, columns=["id", "category", "sensitivity", "arm"])
    df["score"] = scores.numpy()
    df["language"] = lang
    df["resource_tier"] = records[0].resource_tier
    df["model"] = model_id
    df["tag"] = tag

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    RESULTS.commit()

    wide = df.pivot_table(index="id", columns="arm", values="score")
    margin = (wide["nonsyc"] - wide["syc"]).mean()
    syc_rate = (wide["syc"] > wide["nonsyc"]).mean()

    return {
        "lang": lang,
        "n_records": len(records),
        "n_passes": len(items),
        "sycophancy_rate": round(float(syc_rate), 4),
        "mean_margin": round(float(margin), 4),
        "seconds": round(time.time() - t0, 1),
        "passes_per_sec": round(len(items) / (time.time() - t0), 1),
    }


@APP.local_entrypoint()
def main(action: str = "fetch", lang: str = "en", model: str = "", n: int = 300,
         split: str = "eval", source: str = "bench"):
    # The entrypoint runs locally, not in the image.
    sys.path.insert(0, str(SRC))
    from xsyc import constants as K

    if action == "fetch":
        print(json.dumps(fetch.remote([K.MODEL_DEV, K.MODEL_MAIN, K.MODEL_REPLICATION]), indent=2))

    elif action == "smoke":
        # Small model on an L4 with a tiny sample; catches loader, mask and
        # batching faults before any A100 time is spent.
        fn = score_language.with_options(gpu="L4")
        print(json.dumps(fn.remote(lang, K.MODEL_DEV, n=n, tag="smoke"), indent=2))

    elif action == "score":
        model_id = model or K.MODEL_MAIN
        print(json.dumps(
            score_language.remote(lang, model_id, split=split, n=n or None, tag="baseline"),
            indent=2,
        ))

    elif action == "extract":
        print(json.dumps(extract_vectors.remote(model or K.MODEL_MAIN, source=source), indent=2))

    elif action == "sweep":
        rows = sweep_english.remote(model or K.MODEL_MAIN, source=source)
        base = [r for r in rows if r["arm"] == "baseline"][0]
        print(f"\nbaseline margin={base['margin']:+.4f}  syc_rate={base['syc_rate']:.3f}\n")
        print(f"{'arm':10s} {'layer':>5s} {'alpha':>6s} {'margin':>8s} {'d_margin':>9s} {'rel_gain':>9s} {'syc%':>6s}")
        for r in rows:
            if r["arm"] == "baseline":
                continue
            rel = r["d_margin"] / abs(base["margin"]) * 100
            print(f"{r['arm']:10s} {r['layer']:5d} {r['alpha']:6.2f} {r['margin']:+8.4f} "
                  f"{r['d_margin']:+9.4f} {rel:+8.1f}% {r['syc_rate']*100:5.1f}")

    elif action == "transfer":
        mid = model or K.MODEL_MAIN
        tag = "transfer" if mid == K.MODEL_MAIN else "transfer_qwen"
        print(json.dumps(transfer_sweep.remote(K.LANGUAGES, mid, n=n, tag=tag), indent=2))

    elif action == "transfer-argmax":
        # Appendix: v_bench at the grid argmax (alpha 0.8), separate shard dir.
        mid = model or K.MODEL_MAIN
        tag = "transfer_argmax" if mid == K.MODEL_MAIN else "transfer_argmax_qwen"
        print(json.dumps(transfer_sweep.remote(K.LANGUAGES, mid, n=n, tag=tag,
                                               variant="argmax"), indent=2))

    elif action == "aggt":
        r = aggregate_transfer.remote(tag="transfer" if not model else "transfer_qwen")
        order = {"high": 0, "low": 1, "zero": 2}
        per = r["per_language"]
        print(f"languages={r['n_languages']}  conditions={r['conditions']}\n")
        print("TRANSFER RATIO T_l  (1.0 = works as well as in English)")
        for cond in ("v_bench", "v_fact"):
            rows = sorted([x for x in per if x["condition"] == cond],
                          key=lambda x: (order[x["resource_tier"]], -(x["transfer_ratio"] or 0)))
            print(f"\n  [{cond}]")
            for tier in ("high", "low", "zero"):
                sub = [x for x in rows if x["resource_tier"] == tier]
                print(f"    {tier:5s} " + "  ".join(f"{x['language']}:{x['transfer_ratio']:.2f}" for x in sub))
        print("\nBY TIER: mean d_resist and mean transfer ratio")
        print(f"  {'condition':10s} {'tier':6s} {'d_resist':>9s} {'transfer':>9s}")
        for x in sorted(r["by_tier"], key=lambda x: (x["condition"], order[x["resource_tier"]])):
            print(f"  {x['condition']:10s} {x['resource_tier']:6s} {x['d_resist']:+9.4f} {x['transfer']:9.2f}")
        print("\nBY SENSITIVITY: d_resist")
        print(f"  {'condition':10s} {'tier':6s} {'neutral':>9s} {'controv':>9s} {'safety':>9s}")
        bys = r["by_sensitivity"]
        for cond in sorted({x["condition"] for x in bys}):
            for tier in ("high", "low", "zero"):
                vals = {x["sensitivity"]: x["d_resist"] for x in bys
                        if x["condition"] == cond and x["resource_tier"] == tier}
                print(f"  {cond:10s} {tier:6s} " + " ".join(
                    f"{vals.get(s, float('nan')):9.4f}" for s in ("neutral", "controversial", "safety_critical")))

    elif action == "oracle":
        mid = model or K.MODEL_MAIN
        lyr = 16 if mid == K.MODEL_MAIN else 14
        tg = "oracle" if mid == K.MODEL_MAIN else "oracle_qwen"
        print(json.dumps(oracle_sweep.remote(K.LANGUAGES, mid, layer=lyr, n=n, tag=tg), indent=2))

    elif action == "export2":
        import pathlib as _pl
        out = {
            "llama": {"correlates": correlates.remote(K.MODEL_MAIN, tag="oracle"),
                      "ci": bootstrap_ci.remote(tag="transfer")},
            "qwen": {"correlates": correlates.remote(K.MODEL_REPLICATION, tag="oracle_qwen"),
                     "ci": bootstrap_ci.remote(tag="transfer_qwen")},
        }
        dst = _pl.Path(__file__).resolve().parents[1] / "results" / "findings_both.json"
        dst.write_text(json.dumps(out, indent=1, default=float))
        print(f"wrote {dst} ({dst.stat().st_size/1024:.0f} KB)")

    elif action == "export":
        import pathlib as _pl
        out = {
            "correlates": correlates.remote(model or K.MODEL_MAIN),
            "transfer": aggregate_transfer.remote(),
        }
        dst = _pl.Path(__file__).resolve().parents[1] / "results" / "findings.json"
        dst.parent.mkdir(exist_ok=True)
        dst.write_text(json.dumps(out, indent=1, default=float))
        print(f"wrote {dst}  ({dst.stat().st_size/1024:.0f} KB)")

    elif action == "ci":
        r = bootstrap_ci.remote(tag="transfer" if not model else "transfer_qwen")
        order = {"high": 0, "low": 1, "zero": 2}
        rows = sorted(r["rows"], key=lambda x: (order[x["resource_tier"]], -x["v_bench_d"]))
        nb = sum(1 for x in rows if not x.get("beats_random"))
        # No backslashes inside f-string expressions: Python 3.11 rejects them.
        hdr = "v_bench d [95% CI]"
        print(f"\nPaired bootstrap, {r['n_boot']} resamples. 95% CI on d_resist.")
        print(f"{'lang':5s} {'tier':5s} {hdr:>26s} {'vs random':>16s}  sig")
        for x in rows:
            ci = f"{x['v_bench_d']:+.4f} [{x['v_bench_lo']:+.4f},{x['v_bench_hi']:+.4f}]"
            vr = f"[{x['vs_random_lo']:+.4f},{x['vs_random_hi']:+.4f}]"
            print(f"{x['language']:5s} {x['resource_tier']:5s} {ci:>26s} {vr:>16s}  "
                  f"{'yes' if x.get('beats_random') else 'NO'}")
        print(f"\n{nb}/{len(rows)} languages do NOT beat the random control at 95%.")

    elif action == "corr":
        r = correlates.remote(model or K.MODEL_MAIN,
                              tag="oracle" if not model else "oracle_qwen")
        order = {"high": 0, "low": 1, "zero": 2}
        rows = sorted(r["per_language"], key=lambda x: (order[x["resource_tier"]], -x["transfer"]))
        print(f"n = {r['n']} languages\n")
        print(f"{'lang':5s} {'tier':5s} {'transfer':>9s} {'d_bench':>9s} {'d_oracle':>9s} {'cos_en':>7s} {'fert':>6s}")
        for x in rows:
            print(f"{x['language']:5s} {x['resource_tier']:5s} {x['transfer']:9.2f} {x['d_bench']:+9.4f} "
                  f"{x['d_oracle']:+9.4f} {x['cos_en']:7.3f} {x['fertility']:6.1f}")
        print(f"\nORACLE vs ENGLISH VECTOR: oracle wins in {r['oracle_wins']}/38 languages; "
              f"mean d_oracle/d_bench = {r['oracle_ratio_mean']}")
        print(f"\nCORRELATION WITH TRANSFER RATIO (English excluded; n={r['n_excl_en']}, non-independent)")
        print(f"  {'predictor':14s} {'pearson':>9s} {'spearman':>9s} {'(w/ en)':>9s}")
        for k in r["pearson_vs_transfer"]:
            print(f"  {k:14s} {r['pearson_vs_transfer'][k]:9.3f} {r['spearman_vs_transfer'][k]:9.3f}"
                  f" {r['pearson_including_en'][k]:9.3f}")
        print("\nBY TIER")
        print(f"  {'tier':6s} {'transfer':>9s} {'d_bench':>9s} {'d_oracle':>9s} {'cos_en':>7s} {'fert':>6s}")
        for tier in ("high", "low", "zero"):
            b = r["by_tier"][tier]
            print(f"  {tier:6s} {b['transfer']:9.2f} {b['d_bench']:+9.4f} {b['d_oracle']:+9.4f} "
                  f"{b['cos_en']:7.3f} {b['fertility']:6.1f}")

    elif action == "bcnum":
        r = backcheck_numeric.remote()
        print(f"\n{'lang':5s} {'numeric retention':>18s} {'#numbers':>9s}")
        for lg, v in r.items():
            print(f"{lg:5s} {v['numeric_retention']:18.2f} {v['n_numbers']:9d}")
        for lg, v in r.items():
            if v["example_misses"]:
                print(f"\n  {lg} misses:")
                for o, b in v["example_misses"]:
                    print(f"    EN {o}\n    BT {b}")

    elif action == "backcheck":
        rows = backcheck.remote()
        for lang, ex in rows["examples"].items():
            print(f"\n=== {lang}  (numeric-token match {rows['numeric_match'][lang]:.0%})")
            for o, b in ex:
                print(f"  EN  {o}")
                print(f"  BT  {b}")

    elif action == "xprobe":
        print(json.dumps(translate_probe.remote(), indent=2))

    elif action == "xdual":
        rows = dual_stance_multilingual.remote(
            model or K.MODEL_MAIN, layer=n, alpha=0.2 if source == "bench" else 0.4,
            source=source)
        tier = K.RESOURCE_TIER
        print(f"\n[{source}] factual correction, non-English (n=148 propositions/lang)")
        print(f"{'lang':5s} {'tier':5s} {'base':>7s} {'steered':>8s} {'d_wrong':>8s} {'d_right':>8s}")
        for r in sorted(rows, key=lambda x: ({'high':0,'low':1,'zero':2}[tier[x['language']]], x['language'])):
            print(f"{r['language']:5s} {tier[r['language']]:5s} {r['base_wrong']*100:6.1f}% "
                  f"{r['steered_wrong']*100:7.1f}% {r['d_wrong']*100:+7.1f}% {r['d_right']*100:+7.1f}%")

    elif action == "tnorm":
        mid = model or K.MODEL_MAIN
        lyr = 16 if mid == K.MODEL_MAIN else 14
        rows = targetnorm_sweep.remote(K.LANGUAGES, mid, layer=lyr, n=n)
        order = {"high": 0, "low": 1, "zero": 2}
        rows = sorted(rows, key=lambda r: (order[r["resource_tier"]], r["language"]))
        en = [r for r in rows if r["language"] == "en"][0]["d_targetnorm"]
        print(f"\n{'lang':5s} {'tier':5s} {'ref/en':>7s} {'d_targetnorm':>13s} {'T':>6s}")
        for r in rows:
            print(f"{r['language']:5s} {r['resource_tier']:5s} {r['ref_ratio']:7.3f} "
                  f"{r['d_targetnorm']:+13.4f} {r['d_targetnorm'] / en:6.2f}")
        for t in ("high", "low", "zero"):
            sub = [r for r in rows if r["resource_tier"] == t and r["language"] != "en"]
            if not sub:
                continue
            d = sum(r["d_targetnorm"] for r in sub) / len(sub)
            tt = sum(r["d_targetnorm"] / en for r in sub) / len(sub)
            print(f"TIER {t:5s} n={len(sub):2d}  mean d={d:+.4f}  mean T={tt:.3f}")

    elif action == "randbank":
        mid = model or K.MODEL_MAIN
        lyr = 16 if mid == K.MODEL_MAIN else 14
        print(json.dumps(random_bank.remote(mid, layer=lyr, n=n), indent=2))

    elif action == "bankstats":
        r = bank_stats.remote(model or K.MODEL_MAIN)
        order = {"high": 0, "low": 1, "zero": 2}
        rows = sorted(r["rows"], key=lambda x: (order[x["resource_tier"]], -x["d_bench"]))
        print(f"\n{'lang':5s} {'tier':5s} {'d_bench':>9s} {'rand mean':>10s} {'rand sd':>8s} "
              f"{'rand max':>9s} {'pub':>9s} {'#>=':>4s} {'p':>7s} {'p_holm':>7s} sig")
        for x in rows:
            print(f"{x['language']:5s} {x['resource_tier']:5s} {x['d_bench']:+9.4f} "
                  f"{x['rand_mean']:+10.4f} {x['rand_sd']:8.4f} {x['rand_max']:+9.4f} "
                  f"{x['d_published_random']:+9.4f} {x['n_rand_ge_bench']:4d} "
                  f"{x['p_perm']:7.4f} {x['p_holm']:7.4f} {'*' if x['holm_sig'] else '-'}")

    elif action == "ratioci":
        r = ratio_ci.remote(tag="transfer")
        order = {"high": 0, "low": 1, "zero": 2}
        rows = sorted(r["rows"], key=lambda x: (order[x["resource_tier"]], -x["T"]))
        print(f"\nn_ids={r['n_ids']}  n_boot={r['n_boot']}")
        print(f"{'lang':5s} {'tier':5s} {'T':>6s} {'joint 95% CI':>18s} {'naive 95% CI':>18s} {'width x':>8s}")
        for x in rows:
            f = x["joint_width"] / x["naive_width"] if x["naive_width"] else float("nan")
            print(f"{x['language']:5s} {x['resource_tier']:5s} {x['T']:6.3f} "
                  f"[{x['joint_lo']:+7.4f},{x['joint_hi']:+7.4f}] "
                  f"[{x['naive_lo']:+7.4f},{x['naive_hi']:+7.4f}] {f:8.2f}")

    elif action == "heldout":
        r = probe_heldout_ci.remote(model or K.MODEL_MAIN)
        cb, cf = "bench_L16_a0.2", "fact_L8_a0.4"
        print(f"\nexcluding ds_000..ds_{r['excluded_below']-1:03d} (the selection-gate propositions)")
        print(f"{'lang':5s} {'n':>4s} {'vb raw':>22s} {'vb net':>22s} {'vf raw':>22s} {'vf net':>22s}")
        for x in r["rows"]:
            def f(c, t):
                return f"{x[c+'_'+t]*100:+6.1f}[{x[c+'_'+t+'_lo']*100:+6.1f},{x[c+'_'+t+'_hi']*100:+6.1f}]"
            print(f"{x['language']:5s} {x['n']:4d} {x['base_rate']*100:5.1f}% {f(cb,'raw'):>22s} {f(cb,'net'):>22s} "
                  f"{f(cf,'raw'):>22s} {f(cf,'net'):>22s}")

    elif action == "enheld":
        mid = model or K.MODEL_MAIN
        tg = "transfer" if mid == K.MODEL_MAIN else "transfer_qwen"
        print(json.dumps(english_heldout.remote(mid, tag=tg), indent=2))

    elif action == "netci":
        rows = probe_net_ci.remote(model or K.MODEL_MAIN)
        cb, cf = "bench_L16_a0.2", "fact_L8_a0.4"
        print(f"\n{'lang':5s} {'n':>4s} {'v_bench net [95% CI]':>28s} {'v_fact net [95% CI]':>28s}")
        for x in rows:
            print(f"{x['language']:5s} {x['n']:4d} "
                  f"{x[cb+'_net']*100:+9.1f} [{x[cb+'_net_lo']*100:+6.1f},{x[cb+'_net_hi']*100:+6.1f}] "
                  f"{x[cf+'_net']*100:+9.1f} [{x[cf+'_net_lo']*100:+6.1f},{x[cf+'_net_hi']*100:+6.1f}]")

    elif action == "probeci":
        # Both frozen settings in one job, so every probe is scored by the same
        # loaded model.
        specs = [{"source": "bench", "layer": 16, "alpha": 0.2},
                 {"source": "fact", "layer": 8, "alpha": 0.4}]
        r = probe_ci.remote(model or K.MODEL_MAIN, specs)
        cb, cf = "bench_L16_a0.2", "fact_L8_a0.4"
        for cond in ("wrong", "right", "surprising"):
            rows = [x for x in r["rows"] if x["condition"] == cond]
            if not rows:
                continue
            print(f"\n=== {cond}   (n propositions per language shown)")
            print(f"{'lang':5s} {'n':>4s} {'base':>7s} "
                  f"{'v_bench d [95% CI]':>26s} {'v_fact d [95% CI]':>26s}")
            for x in rows:
                print(f"{x['language']:5s} {x['n']:4d} {x['base_rate']*100:6.1f}% "
                      f"{x[cb+'_d']*100:+8.1f} [{x[cb+'_lo']*100:+6.1f},{x[cb+'_hi']*100:+6.1f}] "
                      f"{x[cf+'_d']*100:+8.1f} [{x[cf+'_lo']*100:+6.1f},{x[cf+'_hi']*100:+6.1f}]")

    elif action == "cos":
        print(json.dumps(compare_vectors.remote(model or K.MODEL_MAIN), indent=2))

    elif action == "dual":
        rows = dual_stance.remote(model or K.MODEL_MAIN, layer=n, source=source)
        print(f"\n[{source}]  corrects falsehood / contradicts obvious truth / contradicts surprising truth")
        print(f"{'alpha':>6s} {'wrong':>8s} {'right':>8s} {'surpris':>8s} {'d_wrong':>9s} {'d_surpr':>9s} {'contrarian':>11s}")
        for r in rows:
            c = "     (base)" if r["contrarianism"] is None else f"{r['contrarianism']:11.3f}"
            print(f"{r['alpha']:6.2f} {r['resist_wrong']*100:7.1f}% {r['resist_right']*100:7.1f}% "
                  f"{r['resist_surprising']*100:7.1f}% {r['d_wrong']*100:+8.1f}% {r['d_surprising']*100:+8.1f}% {c}")

    elif action == "agg":
        r = aggregate.remote(tag="baseline")
        order = {"high": 0, "low": 1, "zero": 2}
        rows = sorted(r["by_language"], key=lambda x: (order[x["resource_tier"]], x["syc_rate"]))
        print(f"languages={r['n_languages']}  records={r['n_rows']}\n")
        print("PER-LANGUAGE SYCOPHANCY RATE (%)")
        for tier in ("high", "low", "zero"):
            sub = [x for x in rows if x["resource_tier"] == tier]
            print(f"  [{tier:4s}] " + "  ".join(f"{x['language']}:{x['syc_rate']*100:.0f}" for x in sub))
        paper = {"high": 0.241, "low": 0.315, "zero": 0.352}
        print("\nTIER MEANS")
        for tier in ("high", "low", "zero"):
            o = r["by_tier"][tier]
            print(f"  {tier:5s} ours={o*100:5.1f}  paper={paper[tier]*100:5.1f}  diff={(o-paper[tier])*100:+5.1f}")
        print("\nSYCOPHANCY BY TIER x SENSITIVITY (%)")
        s = r["by_tier_sensitivity"]
        print(f"  {'':6s}" + "".join(f"{c:>18s}" for c in ("neutral", "controversial", "safety_critical")))
        for tier in ("high", "low", "zero"):
            print(f"  {tier:6s}" + "".join(f"{s[c][tier]*100:18.1f}" for c in ("neutral", "controversial", "safety_critical")))

    elif action == "tiers":
        # Reproduce the benchmark paper's published per-tier sycophancy rates
        # for Llama-3.1-8B (high 24.1 / low 31.5 / zero 35.2).
        rows = score_many.remote(K.LANGUAGES, K.MODEL_MAIN, n=n, tag="baseline")
        import collections, statistics
        by = collections.defaultdict(list)
        for r in rows:
            by[r["tier"]].append(r["syc_rate"])
        print(json.dumps(rows, indent=1))
        print("\n=== TIER MEANS (ours vs paper) ===")
        paper = {"high": 0.241, "low": 0.315, "zero": 0.352}
        for tier in ("high", "low", "zero"):
            ours = statistics.mean(by[tier])
            print(f"{tier:5s} n={len(by[tier]):2d}  ours={ours:.3f}  paper={paper[tier]:.3f}  diff={ours-paper[tier]:+.3f}")

    elif action == "repro":
        # Reproduce the published English baseline on the full corpus, to check
        # the scorer against the benchmark authors' S(x, y).
        print(json.dumps(
            score_language.remote(lang, K.MODEL_MAIN, split="all", n=None, tag="repro"),
            indent=2,
        ))

    else:
        raise SystemExit(f"unknown action {action!r}")
