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
