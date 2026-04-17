"""
FastAPI application entry point for the Fake News Detection API.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import (
    API_PREFIX,
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    CORS_ORIGINS,
)
from backend.app.api.routes import router
from backend.app.services import ml_service
from backend.app.services.text_processing import load_spacy_model

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Lifespan — load models on startup
# ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the spaCy NER model and ML pipeline when the server starts."""
    logger.info("🚀 Starting Fake News Detection API v2 …")

    # Load spaCy model first (used by text_processing.clean_text)
    load_spacy_model()

    # Load the trained ML pipeline
    ml_service.load_model()

    yield
    logger.info("👋 Shutting down …")


# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix=API_PREFIX)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect browser root to API docs for convenience."""
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Avoid noisy 404 logs for browser favicon requests."""
    return Response(status_code=204)


# ──────────────────────────────────────────────────────────────
# Convenience runner
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
