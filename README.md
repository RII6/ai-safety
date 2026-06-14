# Internal-State LLM Safety Scanner

A deploy-or-not auditor for fine-tuned LLM checkpoints, aimed at the **ML engineer**
who fine-tuned a model and wants an answer to _"is this safe to ship?"_

Unlike text-level red-teaming tools, which send a prompt
and classify the **response text**, this scanner reads the model's **internal state**
output logits and residual-stream activations. A model can print a polite refusal
while its logits are already primed to comply, text analysis misses that, internal
state does not.

## What it reports

For each checkpoint the scanner emits a verdict with a diagnosis and a recommendation:

| verdict                    | meaning                                                           |
| -------------------------- | ----------------------------------------------------------------- |
| `deploy`                   | refuses harm, represents harm internally — healthy                |
| `deploy_with_restrictions` | partial compliance — ship only with guardrails                    |
| `do_not_deploy`            | complies on harmful prompts                                       |
| `inconclusive`             | the behavioral probe carries no safety signal (e.g. a base model) |

See [`safety_docs.md`](safety_docs.md) for the metrics behind the verdict and the
design decisions.

## Modules

- **`safety_margin`** — behavioral signal. Teacher-forces canonical refuse/comply
  continuations and computes `margin = logP(refuse) − logP(comply)`; `fail_rate` is
  the fraction of harmful prompts where compliance is the cheaper path. This is the
  GCG attack objective (Zou et al. 2023), so it is a validated jailbreakability proxy.
- **`refusal_direction`** — internal-state signal. Difference-of-means "refusal
  direction" (Arditi et al. 2024) in the residual stream; AUROC of how cleanly harmful
  separates from benign measures harm-**awareness**.
- **`verdict`** — aggregates behavior × awareness into the deploy decision.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Runs on **CUDA (NVIDIA, incl. Windows),
Apple Silicon (MPS), or CPU** — the device is auto-detected (CUDA > MPS > CPU), so the
same command works across the team's machines.

```bash
uv sync
```

On Windows with an NVIDIA GPU, the default PyPI `torch` is CPU-only. To use the GPU,
install the CUDA build matching your driver, e.g.:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Build the corpus

The corpus (`data/corpus/{harmful,benign}.jsonl`) ships pre-built. To regenerate or
resize it from public datasets (AdvBench harmful behaviors + Alpaca instructions):

```bash
uv run python build_corpus.py --n 500
```

Both classes are imperative-style, which controls for a prompt-format confound.

## Run

```bash
uv run python main.py                 # full corpus (500/500), full run is slow on CPU
uv run python main.py --sample 30     # fast dev run, 30 prompts per class
uv run python main.py --device cpu    # force a backend (default: auto-detect)
```

`main.py` runs three ground-truth checkpoints — an aligned model, an abliterated one
(safety surgically removed), and a base one (never safety-tuned) — and prints each
module's summary plus the verdict. A correct scanner must separate them.

Windows note: an NVIDIA GPU is picked up automatically as CUDA; without a GPU it falls
back to CPU (correct, just slower — use `--sample` for iteration).
