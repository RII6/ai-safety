import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
REPORTS = ROOT / "reports"

# Per-class prompt cap for a scan. Small so a CPU scan finishes in seconds.
SAMPLE = int(os.environ.get("SCAN_SAMPLE", "25"))

# Reject anything heavier than this before downloading it.
MAX_PARAMS = int(os.environ.get("SCAN_MAX_PARAMS", str(400_000_000)))
MAX_WEIGHT_BYTES = int(os.environ.get("SCAN_MAX_WEIGHT_BYTES", str(1_500_000_000)))

DEVICE = os.environ.get("SCAN_DEVICE", "cpu")

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
DTYPE = _DTYPES[os.environ.get("SCAN_DTYPE", "bfloat16")]

DEMO_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

# ── Dynamic test-case generation (scan-time corpus augmentation) ──────────────
# Off by default. When GEN_N > 0 and an API key is set, each scan mixes in
# GEN_N freshly generated prompts per chosen class on top of the static corpus.
# Generated sets are cached on disk by (provider, class, n, seed) so a scan is
# reproducible and the report-cache key can fold them in. Bump GEN_SEED for a
# new dynamic batch; generation failures fall back to the static corpus.
GEN_N = int(os.environ.get("SCAN_GEN_N", "0"))
GEN_PROVIDER = os.environ.get("SCAN_GEN_PROVIDER", "groq")
GEN_MODEL = os.environ.get("SCAN_GEN_MODEL") or None
GEN_SEED = int(os.environ.get("SCAN_GEN_SEED", "0"))
GEN_CLASS = os.environ.get("SCAN_GEN_CLASS", "harmful")  # harmful | benign | both
GEN_CACHE = CORPUS / "generated"
