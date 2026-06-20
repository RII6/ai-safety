import argparse
import gc
import json
import time

from src.scanner import Model, pick_device, empty_cache
from src.scanner.modules import safety_margin, refusal_direction, verdict, obfuscation
from src.scanner.modules.obfuscation import ObfuscationConfig


def load(path, n=0):
    with open(path, encoding="utf-8") as f:
        prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
    return prompts[:n] if n else prompts


ap = argparse.ArgumentParser()
ap.add_argument("--sample", type=int, default=0, help="per-class prompt cap for fast dev runs (0 = full corpus)")
ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto-detect)")
ap.add_argument("--obfuscation", action="store_true", help="run obfuscation attack battery (slow: ~7x more model calls)")
ap.add_argument("--config", default="configs/general.yaml", help="path to YAML config (default: configs/general.yaml)")
args = ap.parse_args()

device = args.device or pick_device()
harmful = load("data/corpus/harmful.jsonl", args.sample)
benign = load("data/corpus/benign.jsonl", args.sample)
print(f"corpus: {len(harmful)} harmful / {len(benign)} benign | device={device}", flush=True)

CHECKPOINTS = [
    "Qwen/Qwen3-1.7B",
    "huihui-ai/Qwen2.5-1.5B-Instruct-abliterated",
    "Qwen/Qwen2.5-1.5B",
]

for ckpt in CHECKPOINTS:
    print("=" * 70, flush=True)
    print(ckpt, flush=True)
    t0 = time.time()
    model = Model(ckpt, device)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    margin = safety_margin.run(model, harmful, benign)
    print(f"  affirmative_margin done in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    direction = refusal_direction.run(model, harmful, benign)
    print(f"  refusal_direction done in {time.time() - t0:.1f}s", flush=True)

    report = verdict.compute(margin, direction)

    print("[affirmative_margin]", json.dumps(margin["summary"], indent=2), flush=True)
    print("[refusal_direction] ", json.dumps(direction["summary"], indent=2), flush=True)
    print("[verdict]           ", json.dumps(report["summary"], indent=2), flush=True)

    if args.obfuscation:
        obf_cfg = ObfuscationConfig.from_yaml(args.config)
        t0 = time.time()
        obf_result = obfuscation.run(model, harmful, config=obf_cfg)
        print(f"  obfuscation done in {time.time() - t0:.1f}s", flush=True)
        print("[obfuscation]       ", json.dumps(obf_result["summary"], indent=2), flush=True)
    print(flush=True)

    del model
    gc.collect()
    empty_cache(device)
