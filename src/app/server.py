from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db
from .scan import ScanError, scan

STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="LLM Safety Scanner", lifespan=lifespan)


class ScanRequest(BaseModel):
    repo: str
    force: bool = False


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "demo_model": config.DEMO_MODEL,
        "sample": config.SAMPLE,
        "device": config.DEVICE,
    }


@app.post("/api/scan")
def run_scan(req: ScanRequest):
    try:
        return scan(req.repo, force=req.force)
    except ScanError as e:
        return JSONResponse(status_code=e.status, content={"error": e.message})


@app.get("/api/reports")
def list_reports():
    return db.list_scans()


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
