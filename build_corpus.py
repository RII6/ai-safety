"""Build the harmful/benign corpus from public datasets.

Scales the hand-written seed corpus (~20 each) up to a real evaluation set by
pulling established safety datasets and merging them with whatever is already in
data/corpus/. Non-destructive: existing curated prompts are kept and deduped
against the new ones.

Sources:
  harmful  <- walledai/AdvBench   (the canonical jailbreak/GCG behavior set)
  benign   <- tatsu-lab/alpaca    (standalone instructions, input-free)

Both are imperative-style ("Write...", "Explain...", "Give..."), which controls
for the prompt-format confound that contaminated the earlier questions-vs-
imperatives split (see the refusal-direction / coupling notes).

Usage:
    uv run python build_corpus.py            # default: 500 per class
    uv run python build_corpus.py --n 1000
"""
import argparse
import csv
import io
import json
import random
import urllib.request
from pathlib import Path

from datasets import load_dataset

# Canonical AdvBench behaviors from the GCG repo (ungated; the HF mirror is gated).
ADVBENCH_CSV = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"

CORPUS = Path("data/corpus")
HARMFUL = CORPUS / "harmful.jsonl"
BENIGN = CORPUS / "benign.jsonl"


def _read_existing(path):
    if not path.exists():
        return []
    return [json.loads(line)["prompt"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm(p):
    return " ".join(p.lower().split())


def _dedup(prompts):
    seen, out = set(), []
    for p in prompts:
        p = p.strip()
        key = _norm(p)
        if p and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def load_advbench():
    with urllib.request.urlopen(ADVBENCH_CSV) as resp:
        text = resp.read().decode("utf-8")
    rows = csv.DictReader(io.StringIO(text))
    return [r["goal"] for r in rows]


def load_alpaca():
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    # Keep only standalone instructions (no separate input the prompt refers to).
    return [r["instruction"] for r in ds if not r["input"].strip()]


def build(path, fetched, n, seed):
    existing = _read_existing(path)
    pool = _dedup(existing + fetched)
    # Existing curated prompts always survive; fill the rest from the dataset.
    keep = existing + [p for p in pool if p not in existing]
    rng = random.Random(seed)
    rng.shuffle(keep)
    if n:
        keep = keep[:n]
    path.write_text("".join(json.dumps({"prompt": p}, ensure_ascii=False) + "\n" for p in keep), encoding="utf-8")
    return len(existing), len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="prompts per class (0 = all available)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    CORPUS.mkdir(parents=True, exist_ok=True)

    print("fetching AdvBench (harmful)...", flush=True)
    h_old, h_new = build(HARMFUL, load_advbench(), args.n, args.seed)
    print(f"  harmful: {h_old} seed -> {h_new} total -> {HARMFUL}", flush=True)

    print("fetching Alpaca (benign)...", flush=True)
    b_old, b_new = build(BENIGN, load_alpaca(), args.n, args.seed)
    print(f"  benign:  {b_old} seed -> {b_new} total -> {BENIGN}", flush=True)


if __name__ == "__main__":
    main()
