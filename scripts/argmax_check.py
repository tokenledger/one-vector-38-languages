"""Appendix check: v_bench at the grid argmax (alpha 0.8) versus the frozen setting.

Reads the transfer shards under modal_backup/xsyc-results/{transfer,
transfer_argmax, transfer_qwen, transfer_argmax_qwen} and writes
results/argmax_check.json. English is excluded from every tier mean, as in the
paper. Run from the repository root:

    python scripts/argmax_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xsyc import constants as K  # noqa: E402

SHARDS = ROOT / "modal_backup" / "xsyc-results"
RUNS = {
    "llama": ("Llama-3.1-8B-Instruct",
              {"frozen": ("transfer", "L16 a0.2"), "argmax": ("transfer_argmax", "L16 a0.8")}),
    "qwen": ("Qwen2.5-7B-Instruct",
             {"frozen": ("transfer_qwen", "L14 a0.2"), "argmax": ("transfer_argmax_qwen", "L21 a0.8")}),
}


def _wide(df: pd.DataFrame, cond: str) -> pd.DataFrame:
    return df[df.condition == cond].pivot_table(index="id", columns="arm", values="score")


def margin(df: pd.DataFrame, cond: str) -> float:
    w = _wide(df, cond)
    return float((w["nonsyc"] - w["syc"]).mean())


def syc_rate(df: pd.DataFrame, cond: str) -> float:
    w = _wide(df, cond)
    return float((w["syc"] > w["nonsyc"]).mean())


def summarise(short: str, tag: str) -> dict:
    sh = {lg: pd.read_parquet(SHARDS / tag / f"{short}__{lg}.parquet") for lg in K.LANGUAGES}
    en = sh["en"]
    base_en = margin(en, "none")
    en_d = margin(en, "v_bench") - base_en
    rows = {}
    for lg, df in sh.items():
        base = margin(df, "none")
        d = margin(df, "v_bench") - base
        r = margin(df, "random") - base
        rows[lg] = {"d_resist": d, "d_random": r, "transfer": d / en_d,
                    "tier": K.RESOURCE_TIER[lg]}
    non_en = {lg: v for lg, v in rows.items() if lg != "en"}
    by_tier = {
        t: {k: float(np.mean([v[k] for v in non_en.values() if v["tier"] == t]))
            for k in ("transfer", "d_resist", "d_random")}
        for t in ("high", "low", "zero")
    }
    return {
        "english": {"d_resist": en_d, "rel_gain_pct": en_d / base_en * 100,
                    "d_random": margin(en, "random") - base_en,
                    "syc_base": syc_rate(en, "none"), "syc_steered": syc_rate(en, "v_bench")},
        "by_tier_excl_en": by_tier,
        "beats_random_of_37": int(sum(v["d_resist"] > v["d_random"] for v in non_en.values())),
        "per_language": rows,
    }


def main() -> None:
    out = {}
    for model, (short, runs) in RUNS.items():
        out[model] = {}
        for name, (tag, setting) in runs.items():
            out[model][name] = {"setting": setting, "tag": tag, **summarise(short, tag)}
        f, a = out[model]["frozen"]["per_language"], out[model]["argmax"]["per_language"]
        langs = [lg for lg in K.LANGUAGES if lg != "en"]
        out[model]["per_language_T_correlation"] = float(np.corrcoef(
            [f[lg]["transfer"] for lg in langs], [a[lg]["transfer"] for lg in langs])[0, 1])
    (ROOT / "results" / "argmax_check.json").write_text(json.dumps(out, indent=1))
    for model, m in out.items():
        for name in ("frozen", "argmax"):
            r = m[name]
            print(f"{model} {name} ({r['setting']}): en {r['english']['rel_gain_pct']:+.1f}%  "
                  + "  ".join(f"{t} T={v['transfer']:.3f}" for t, v in r["by_tier_excl_en"].items())
                  + f"  beats random {r['beats_random_of_37']}/37")
        print(f"{model} per-language T correlation frozen vs argmax: {m['per_language_T_correlation']:.3f}")


if __name__ == "__main__":
    main()
