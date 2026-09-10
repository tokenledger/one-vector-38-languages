# One Vector, Thirty-Eight Languages

<https://github.com/tokenledger/one-vector-38-languages>

Code, vectors, and per-item results for *One Vector, Thirty-Eight Languages:
Benchmark-Derived Anti-Sycophancy Steering Works Worst Where Sycophancy Is
Worst* (ORACLE 2026). An anti-sycophancy steering direction extracted from
English data alone is applied unchanged across the 38 languages of a parallel
forced-choice benchmark, on Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, with
a reduced check on Llama-3.1-70B-Instruct.

## What is released

| What | Where |
|---|---|
| Scorer, extraction, splits, probes | `src/xsyc/` |
| Every GPU experiment (Modal) | `modal_apps/app.py` |
| 70B check (NDIF / nnsight) | `scripts/scale_check.py`, `src/xsyc/ndif70b.py` |
| Figures | `scripts/make_figures.py` → `figures/fig1..5.pdf` (generated, not committed) |
| English vectors `v_bench`, `v_fact` (both models) | `modal_backup/xsyc-results/vectors/` |
| 76 target-language vectors (38 per model) | `modal_backup/xsyc-results/vectors/oracle__*.pt` |
| Per-item benchmark scores, all conditions | `modal_backup/xsyc-results/transfer*/`, `oracle*/`, `randbank/` |
| Probe scores, translated probe, intervals | `modal_backup/xsyc-results/probe_i18n/`, `sweep/dualstance*` |
| Aggregates the paper reads | `results/*.json`, `results/scale70b/` |

The paper source and the working notes behind it live in a separate
repository. In the code and in result file names, "oracle" means the paper's
target-language vector: a vector extracted from the target language's own
extract split.

The benchmark itself is `aryashah00/multilingual-sycophancy` on the Hugging
Face Hub and is not redistributed here; the Modal app downloads it. Model
weights are gated (Llama) or public (Qwen) under their own licences.

## Install

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"         # scorer + tests, CPU only
pip install -e ".[plot]"         # figures
pip install -e ".[modal]"        # GPU experiments
pip install -e ".[ndif]"         # 70B check
```

Python 3.10+. The GPU image pins `torch==2.6.0`, `transformers==4.51.3`,
`accelerate==1.6.0` (see `modal_apps/app.py`); local installs float.

## Reproduce

Six CPU tests, about five seconds, including batched-vs-unbatched scoring
equivalence and steering-hook cleanup:

```
pytest -q tests
```

Figures from the committed aggregates (no GPU); writes `figures/fig1.pdf`
through `figures/fig5.pdf`, creating the directory if needed:

```
python scripts/make_figures.py
```

The full study (roughly 25 A100-40GB hours across both models, ~2 hours per
38-language sweep) runs on Modal with `HF_TOKEN` exported:

```
modal run modal_apps/app.py --action fetch
modal run modal_apps/app.py --action extract   --model meta-llama/Llama-3.1-8B-Instruct
modal run modal_apps/app.py --action sweep     --model meta-llama/Llama-3.1-8B-Instruct
modal run modal_apps/app.py --action transfer  --model meta-llama/Llama-3.1-8B-Instruct
modal run modal_apps/app.py --action aggregate
```

The remaining actions (dual-stance probe, target-language vectors, random
bank, bootstrap and ratio intervals, translated probe) are dispatched by name
in `main` at the bottom of `modal_apps/app.py`. The 70B check is
`python scripts/scale_check.py extract|sweep|transfer|report` with
`NDIF_API_KEY` exported.

## Fixed quantities

| Quantity | Value | Source |
|---|---|---|
| Benchmark | `aryashah00/multilingual-sycophancy`, 4,950 records × 38 languages | `src/xsyc/constants.py` |
| Split seed | `0xA71`; 396 extract / 297 dev / 4,257 eval, ID-disjoint, shared across languages | `constants.py`, `data.py` |
| Evaluation sample | 300 IDs, sensitivity-balanced 100/100/100, drawn once before any vector | `data.py` |
| Llama-3.1-8B frozen | `v_bench` L16 α0.2; `v_fact` L8 α0.4 | `modal_apps/app.py::conditions_for` |
| Qwen2.5-7B frozen | `v_bench` L14 α0.2; `v_fact` L7 α0.2 | same |
| Llama-3.1-70B | L20 α0.4, argmax over α≤0.4 (different rule; see paper App. A) | `results/scale70b/selection.json` |
| Grid-maximum check | `v_bench` Llama L16 α0.8, Qwen L21 α0.8; `modal run modal_apps/app.py --action transfer-argmax`; `python scripts/argmax_check.py` → `results/argmax_check.json` | paper App. B |
| Random control | seed 0 at the `v_bench` layer; bank seeds 100–119 | `app.py` |
| Bootstrap | 4,000 paired resamples, seed 0 | `app.py::bootstrap_ci`, `ratio_ci` |
| Model revisions | Hub `main` at download time (July–August 2026); not pinned by hash | `app.py::fetch` |

## Selection, stated plainly

`v_bench`'s alpha was held at 0.2 on both models after the development sweep
showed no saturation through 0.8; only the layer was chosen by the margin
criterion, at that alpha. It is not the argmax over the grid. `v_fact` on Llama
is the best-margin setting passing a probe gate; on Qwen it is a carried-over
depth with the default alpha. The development sweeps these were read from are
in `modal_backup/xsyc-results/sweep/`.

## Manifest

`results/MANIFEST.sha256` lists the checksum of every file under `results/`
and `modal_backup/` and the commit that produced the manifest. Verify from the
repository root with `shasum -a 256 -c results/MANIFEST.sha256`.

## Citation

See `CITATION.cff`.

## Licence

Code is MIT (`LICENSE`). Vectors and per-item results are released under the
same terms. The benchmark and model weights remain under their own licences.
