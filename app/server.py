from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import json

from . import config
from .scan import ScanError, scan

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="LLM Safety Scanner")


class ScanRequest(BaseModel):
    repo: str
    force: bool = False


@app.get("/api/health")
def health():
    return {"ok": True, "demo_model": config.DEMO_MODEL, "sample": config.SAMPLE}


@app.post("/api/scan")
def run_scan(req: ScanRequest):
    try:
        return scan(req.repo, force=req.force)
    except ScanError as e:
        return JSONResponse(status_code=e.status, content={"error": e.message})


@app.get("/api/reports")
def list_reports():
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(config.REPORTS.glob("*.json")):
        rep = json.loads(p.read_text(encoding="utf-8"))
        out.append({"repo": rep["repo"], "verdict": rep["verdict"]["code"]})
    return out


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
