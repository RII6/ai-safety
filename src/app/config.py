

import os
from pathlib import Path

import torch

from src.scanner.inference import pick_device

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
REPORTS = ROOT.parent / "reports"

_CONFIG_PATH = Path(os.environ.get("SCAN_CONFIG", ROOT / "configs" / "general.yaml"))


def _load_section(name: str) -> dict:

    try:
        import yaml

        with open(_CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return raw.get(name, {}) or {}
    except Exception:
        return {}


_cfg = _load_section("scanner")
_gen = _cfg.get("generation", {}) or {}


def _resolve(env_key: str, source: dict, key: str, default, cast=str):
    if env_key in os.environ:
        return cast(os.environ[env_key])
    if source.get(key) is not None:
        return cast(source[key])
    return default


SAMPLE = _resolve("SCAN_SAMPLE", _cfg, "sample", 25, int)
MAX_PARAMS = _resolve("SCAN_MAX_PARAMS", _cfg, "max_params", 400_000_000, int)
MAX_WEIGHT_BYTES = _resolve("SCAN_MAX_WEIGHT_BYTES", _cfg, "max_weight_bytes", 1_500_000_000, int)
def _resolve_device() -> str:
    raw = _resolve("SCAN_DEVICE", _cfg, "device", "auto", str)
    return pick_device() if raw == "auto" else raw


DEVICE = _resolve_device()

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
DTYPE = _DTYPES[_resolve("SCAN_DTYPE", _cfg, "dtype", "bfloat16", str)]

DEMO_MODEL = _resolve("SCAN_DEMO_MODEL", _cfg, "demo_model", "HuggingFaceTB/SmolLM2-360M-Instruct", str)

DATABASE_URL = _resolve(
    "DATABASE_URL", _cfg, "database_url", "postgresql://postgres:postgres@localhost:5432/aisafety", str
)

GEN_N = _resolve("SCAN_GEN_N", _gen, "n", 0, int)
GEN_PROVIDER = _resolve("SCAN_GEN_PROVIDER", _gen, "provider", "groq", str)
GEN_MODEL = _resolve("SCAN_GEN_MODEL", _gen, "model", None, str) or None
GEN_SEED = _resolve("SCAN_GEN_SEED", _gen, "seed", 0, int)
GEN_CLASS = _resolve("SCAN_GEN_CLASS", _gen, "class", "harmful", str)
GEN_CACHE = CORPUS / "generated"
