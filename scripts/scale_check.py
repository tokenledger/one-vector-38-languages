"""Llama-3.1-70B scale check driver. Runs remotely on NDIF.

Mirrors the main study's protocol so the numbers are comparable (with one
difference, see below):
  1. extract v_bench from the English `extract` split at 25/50/75 percent depth
  2. select layer and alpha on the English `dev` split only, then freeze
  3. transfer to 10 languages, conditions none / v_bench / random matched-norm

Scoped down from the full study on purpose: 10 languages not 38, 150 items per
language not 300, and no reversed or v_fact arm. Round trips are the cost here,
not compute.

Selection differs from the primary models: this driver sweeps alpha only
through 0.4 and takes the argmax of dev margin gain over the grid, whereas the
7-8B models held alpha at 0.2 and chose only the layer. The 70B setting (L20,
alpha 0.4) is therefore at twice the relative strength.

Checkpoints after every stage and every language into results/scale70b/.

Usage:
    python scripts/scale_check.py extract
    python scripts/scale_check.py sweep
    python scripts/scale_check.py transfer
    python scripts/scale_check.py report
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xsyc import constants as K  # noqa: E402
from xsyc import data as D  # noqa: E402
from xsyc.ndif70b import (  # noqa: E402
    LAYERS,
    batches,
    collect_remote,
    connect,
    pad_batch,
    score_remote,
    with_retry,
)
from xsyc.scoring import difference_vector, encode_items, unit  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results" / "scale70b"
OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["en", "es", "de", "zh", "ar", "hi", "id", "bn", "te", "my"]
N_PER_LANG = 150
ALPHAS = [0.05, 0.1, 0.2, 0.4]


def _items(records):
    """Interleave (prompt, sycophantic) and (prompt, non-sycophantic)."""
    out = []
    for r in records:
        out.append((r.prompt, r.sycophantic))
        out.append((r.prompt, r.not_sycophantic))
    return out


def _score_all(lm, tok, items, steer=None, tag=""):
    """Score every item, batching under the token budget. Returns (n,) tensor."""
    enc = encode_items(tok, items)
    scores = torch.full((len(items),), float("nan"))
    t0 = time.time()
    n_traces = 0
    for idx in batches(enc):
        batch = pad_batch([enc[i] for i in idx], tok.pad_token_id)
        scores[idx] = with_retry(score_remote, lm, batch, steer=steer)
        n_traces += 1
    assert not torch.isnan(scores).any(), "some items were never scored"
    dt = time.time() - t0
    if tag:
        print(
            f"    {tag}: {len(items)} seqs, {n_traces} traces, "
            f"{dt:.0f}s -> {len(items) / dt:.1f} seq/s",
            flush=True,
        )
    return scores


def _margin(scores):
    """Items are interleaved syc/nonsyc, so even indices are sycophantic."""
    syc = scores[0::2]
    non = scores[1::2]
    return float((non - syc).mean()), float((syc > non).float().mean())


# --- stage 1: extraction ---------------------------------------------------


def stage_extract():
    lm = connect()
    tok = lm.tokenizer
    recs = D.take(D.load_language("en", DATA), "extract")
    print(f"extracting from {len(recs)} English pairs at layers {LAYERS}")

    syc = [(r.prompt, r.sycophantic) for r in recs]
    non = [(r.prompt, r.not_sycophantic) for r in recs]

    def gather(pairs, label):
        enc = encode_items(tok, pairs)
        acc = {lyr: [] for lyr in LAYERS}
        t0 = time.time()
        for idx in batches(enc):
            batch = pad_batch([enc[i] for i in idx], tok.pad_token_id)
            got = with_retry(collect_remote, lm, batch, LAYERS)
            for lyr in LAYERS:
                acc[lyr].append(got[lyr])
        print(f"  {label}: {len(pairs)} seqs in {time.time() - t0:.0f}s", flush=True)
        return {lyr: torch.cat(v) for lyr, v in acc.items()}

    a_syc = gather(syc, "sycophantic")
    a_non = gather(non, "non-sycophantic")

    blob = {"model": "70B", "layers": LAYERS}
    summary = {}
    for lyr in LAYERS:
        v = difference_vector(a_non[lyr], a_syc[lyr])
        ref = float(torch.cat([a_syc[lyr], a_non[lyr]]).norm(dim=-1).mean())
        blob[lyr] = {"vector": v, "ref_norm": ref}
        cos = torch.nn.functional.cosine_similarity(
            a_non[lyr].mean(0), a_syc[lyr].mean(0), dim=0
        )
        summary[lyr] = {
            "v_norm": round(float(v.norm()), 4),
            "ref_norm": round(ref, 4),
            "rel": round(float(v.norm()) / ref, 4),
            "class_mean_cos": round(float(cos), 4),
        }
    torch.save(blob, OUT / "v_bench_70b.pt")
    (OUT / "extract_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


# --- stage 2: English-only selection --------------------------------------


def stage_sweep():
    lm = connect()
    tok = lm.tokenizer
    blob = torch.load(OUT / "v_bench_70b.pt", weights_only=False)
    recs = D.take(D.load_language("en", DATA), "dev")
    items = _items(recs)
    print(f"English dev sweep: {len(recs)} records, {len(items)} sequences")

    sweep_path = OUT / "sweep.json"
    rows = json.loads(sweep_path.read_text()) if sweep_path.exists() else []
    done = {(r.get("layer"), r.get("alpha"), r["arm"]) for r in rows}

    if not any(r["arm"] == "baseline" for r in rows):
        base = _score_all(lm, tok, items, tag="baseline")
        b_margin, b_syc = _margin(base)
        rows.append({"arm": "baseline", "layer": None, "alpha": 0.0,
                     "margin": round(b_margin, 5), "syc_rate": round(b_syc, 4)})
        sweep_path.write_text(json.dumps(rows, indent=1))
    else:
        b = next(r for r in rows if r["arm"] == "baseline")
        b_margin, b_syc = b["margin"], b["syc_rate"]
    print(f"  baseline margin={b_margin:+.4f} syc_rate={b_syc:.3f}", flush=True)

    for lyr in LAYERS:
        v, ref = blob[lyr]["vector"], blob[lyr]["ref_norm"]
        for a in ALPHAS:
            if (lyr, a, "v_bench") in done:
                continue
            delta = unit(v) * ref * a
            s = _score_all(lm, tok, items, steer=(lyr, delta))
            m, sr = _margin(s)
            gain = (m - b_margin) / abs(b_margin) * 100
            rows.append({"arm": "v_bench", "layer": lyr, "alpha": a,
                         "margin": round(m, 5), "syc_rate": round(sr, 4),
                         "d_margin": round(m - b_margin, 5), "rel_gain": round(gain, 1)})
            print(f"  L{lyr} a={a}: margin={m:+.4f} ({gain:+.1f}%) syc={sr:.3f}", flush=True)
            (OUT / "sweep.json").write_text(json.dumps(rows, indent=1))

    # Matched-norm random control at the selected setting.
    best = max((r for r in rows if r["arm"] == "v_bench"), key=lambda r: r["d_margin"])
    if (best["layer"], best["alpha"], "random") in done:
        print("  random control already recorded")
        (OUT / "selection.json").write_text(
            json.dumps({"layer": best["layer"], "alpha": best["alpha"]}, indent=1))
        print(f"\nSELECTED (English only): layer {best['layer']}, alpha {best['alpha']}")
        return
    torch.manual_seed(0)
    v = blob[best["layer"]]["vector"]
    rnd = unit(torch.randn_like(v)) * blob[best["layer"]]["ref_norm"] * best["alpha"]
    s = _score_all(lm, tok, items, steer=(best["layer"], rnd))
    m, sr = _margin(s)
    rows.append({"arm": "random", "layer": best["layer"], "alpha": best["alpha"],
                 "margin": round(m, 5), "syc_rate": round(sr, 4),
                 "d_margin": round(m - b_margin, 5),
                 "rel_gain": round((m - b_margin) / abs(b_margin) * 100, 1)})
    print(f"  random L{best['layer']} a={best['alpha']}: "
          f"{(m - b_margin) / abs(b_margin) * 100:+.1f}%", flush=True)

    (OUT / "sweep.json").write_text(json.dumps(rows, indent=1))
    (OUT / "selection.json").write_text(
        json.dumps({"layer": best["layer"], "alpha": best["alpha"]}, indent=1)
    )
    print(f"\nSELECTED (English only): layer {best['layer']}, alpha {best['alpha']}")


# --- stage 3: transfer -----------------------------------------------------


def stage_transfer():
    lm = connect()
    tok = lm.tokenizer
    blob = torch.load(OUT / "v_bench_70b.pt", weights_only=False)
    sel = json.loads((OUT / "selection.json").read_text())
    lyr, alpha = sel["layer"], sel["alpha"]
    v, ref = blob[lyr]["vector"], blob[lyr]["ref_norm"]
    delta = unit(v) * ref * alpha
    torch.manual_seed(0)
    rnd = unit(torch.randn_like(v)) * ref * alpha

    print(f"transfer at layer {lyr}, alpha {alpha}, {len(LANGS)} languages")
    for lang in LANGS:
        path = OUT / f"transfer_{lang}.json"
        if path.exists():
            print(f"  {lang}: cached")
            continue
        recs = D.stratified_sample(D.take(D.load_language(lang, DATA), "eval"), N_PER_LANG)
        items = _items(recs)
        row = {"language": lang, "tier": recs[0].resource_tier, "n": len(recs)}
        for name, st in (("none", None), ("v_bench", (lyr, delta)), ("random", (lyr, rnd))):
            s = _score_all(lm, tok, items, steer=st, tag=f"{lang}/{name}")
            m, sr = _margin(s)
            row[f"{name}_margin"] = round(m, 5)
            row[f"{name}_syc"] = round(sr, 4)
        row["d_bench"] = round(row["v_bench_margin"] - row["none_margin"], 5)
        row["d_random"] = round(row["random_margin"] - row["none_margin"], 5)
        path.write_text(json.dumps(row, indent=1))
        print(f"  {lang} [{row['tier']}]: d_bench={row['d_bench']:+.5f} "
              f"d_random={row['d_random']:+.5f}", flush=True)


# --- stage 4: report -------------------------------------------------------


def stage_report():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("transfer_*.json"))]
    if not rows:
        raise SystemExit("no transfer results yet")
    en = next(r for r in rows if r["language"] == "en")
    en_d = en["d_bench"]

    for r in rows:
        r["transfer"] = round(r["d_bench"] / en_d, 3) if en_d else None

    order = {"high": 0, "low": 1, "zero": 2}
    rows.sort(key=lambda r: (order[r["tier"]], -r["d_bench"]))

    print(f"\n70B English baseline: margin {en['none_margin']:+.4f}, "
          f"sycophancy {en['none_syc'] * 100:.1f}%")
    print(f"70B English Delta_resist: {en_d:+.5f}\n")
    print(f"{'lang':5s} {'tier':5s} {'base_syc':>9s} {'d_bench':>9s} {'d_random':>9s} {'T':>7s}")
    for r in rows:
        print(f"{r['language']:5s} {r['tier']:5s} {r['none_syc'] * 100:8.1f}% "
              f"{r['d_bench']:+9.5f} {r['d_random']:+9.5f} {r['transfer']:7.2f}")

    print("\nBY TIER (mean transfer ratio)")
    print(f"  {'tier':6s} {'70B':>7s}   {'8B':>7s} {'Qwen7B':>8s}")
    ref8 = {"high": 0.51, "low": 0.19, "zero": 0.03}
    refq = {"high": 0.33, "low": 0.07, "zero": 0.01}
    tiers = {}
    for t in ("high", "low", "zero"):
        sub = [r for r in rows if r["tier"] == t]
        if not sub:
            continue
        mean_t = sum(r["transfer"] for r in sub) / len(sub)
        tiers[t] = round(mean_t, 3)
        print(f"  {t:6s} {mean_t:7.2f}   {ref8[t]:7.2f} {refq[t]:8.2f}")
    (OUT / "summary.json").write_text(
        json.dumps({"english": en, "by_tier": tiers, "rows": rows}, indent=1)
    )


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"extract": stage_extract, "sweep": stage_sweep,
     "transfer": stage_transfer, "report": stage_report}[stage]()
