"""
FastAPI Main Application for Jindal Stainless Steel Defect Detection Engine.
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
import yaml
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database.database import init_db
from backend.model_manager import ModelManager
from backend.routes.inspect import router as inspect_router
from backend.routes.analytics import router as analytics_router
from backend.routes.models import router as models_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend")


def load_app_config() -> dict:
    config_path = ROOT / "configs" / "config.yaml"
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using empty config.")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Config, Initialize SQLite DB, Preload AI Models
    logger.info("Initializing Jindal AI Defect Detection Backend...")
    config = load_app_config()

    db_cfg = config.get("database", {})
    db_path = db_cfg.get("path", "database/inspections.db")
    abs_db_path = ROOT / db_path
    init_db(str(abs_db_path))

    # Preload Model Manager
    mm = ModelManager(config)
    mm.load_all_models()

    logger.info("AI Models preloaded and ready for inference.")
    yield
    # Shutdown
    logger.info("Shutting down Jindal AI Inspection Backend.")


app = FastAPI(
    title="Jindal Stainless AI Defect Detection API",
    description="Real-Time Steel Surface Inspection, Multi-Model Ensemble, and PLC Integration Engine",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware for web dashboards and industrial HMI clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(inspect_router)
app.include_router(analytics_router)
app.include_router(models_router)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Jindal Stainless AI Defect Detection Engine",
        "version": "2.0.0",
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start_server()
