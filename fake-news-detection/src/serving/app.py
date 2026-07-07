"""
app.py — FastAPI serving layer for the fake news detection model.

Loads the trained sklearn pipeline (TF-IDF + LogisticRegression) at module
level and exposes /predict, /health, and /model/version endpoints.

CRITICAL DESIGN DECISION — No train/serve skew:
    Training (train_baseline.py) fitted the pipeline on df_train["clean_text"],
    which is the text column after applying clean_text(). This serving layer
    reuses the exact same clean_text function from src.preprocessing.clean_text.
    The title field is accepted for forward compatibility but NOT concatenated
    with text — the v0.1 baseline was trained on text-only.
"""

import json
import logging
import traceback
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from src.preprocessing.clean_text import clean_text
from src.serving.schemas import (
    HealthResponse,
    ModelVersionResponse,
    PredictRequest,
    PredictResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Paths (relative to project root, where uvicorn is launched)
# ---------------------------------------------------------------------------
_MODEL_PATH = Path("models/v0.1_baseline.joblib")
_METRICS_PATH = Path("models/v0.1_baseline_metrics.json")
_MODEL_VERSION = "v0.1_baseline"

# ---------------------------------------------------------------------------
# Minimum token count for OOD warning (heuristic, not a hard block)
# ---------------------------------------------------------------------------
_MIN_TOKEN_COUNT = 3

# ---------------------------------------------------------------------------
# Load model and metrics at module level (once, not per request)
# ---------------------------------------------------------------------------
_pipeline = None
_metrics: dict | None = None
model_loaded: bool = False

try:
    _pipeline = joblib.load(_MODEL_PATH)
    logger.info("Model loaded successfully from %s", _MODEL_PATH)
except Exception as exc:
    logger.error(
        "Failed to load model from %s: %s\n%s",
        _MODEL_PATH,
        exc,
        traceback.format_exc(),
    )

try:
    with open(_METRICS_PATH, "r") as f:
        _metrics = json.load(f)
    logger.info("Metrics loaded successfully from %s", _METRICS_PATH)
except Exception as exc:
    logger.error(
        "Failed to load metrics from %s: %s\n%s",
        _METRICS_PATH,
        exc,
        traceback.format_exc(),
    )

# Model is "loaded" only if both the pipeline and metrics loaded successfully
model_loaded = _pipeline is not None and _metrics is not None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FactRadar — Fake News Detection API",
    version=_MODEL_VERSION,
    description="Serves the baseline TF-IDF + Logistic Regression model.",
)


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Classify a single article as fake or real.

    Edge cases handled:
    - Model not loaded → 503
    - Cleaned text is empty (input was purely URLs/punctuation) → 422
    - Cleaned text has < 3 tokens → 200 with warning="low_confidence_ood"
    - Unexpected exception → 500 (logged server-side, generic message returned)
    """
    # 1. Model availability check
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Service is degraded.",
        )

    try:
        # 2. Apply the SAME clean_text used at training time.
        #    Use text field only — v0.1 baseline was trained on text-only.
        cleaned = clean_text(request.text)

        # 3. Edge case: input cleaned to empty string
        if not cleaned:
            raise HTTPException(
                status_code=422,
                detail="Input contains no usable text content after preprocessing.",
            )

        # 4. OOD warning for very short inputs
        warning = None
        token_count = len(cleaned.split())
        if token_count < _MIN_TOKEN_COUNT:
            warning = "low_confidence_ood"

        # 5. Run prediction
        #    sklearn pipeline expects a list/array of strings (2D-like input).
        probabilities = _pipeline.predict_proba([cleaned])[0]
        # classes_ gives the label order: typically [0, 1] where 0=real, 1=fake
        classes = _pipeline.classes_
        predicted_class_idx = probabilities.argmax()
        confidence = float(probabilities[predicted_class_idx])
        predicted_class = int(classes[predicted_class_idx])

        # Map numeric label to string: 0 → "real", 1 → "fake"
        predicted_label = "fake" if predicted_class == 1 else "real"

        return PredictResponse(
            predicted_label=predicted_label,
            confidence=round(confidence, 6),
            model_version=_MODEL_VERSION,
            warning=warning,
        )

    except HTTPException:
        # Re-raise HTTP exceptions (422 from empty-after-clean check)
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error during prediction: %s\n%s",
            exc,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction.",
        )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check reflecting whether the model loaded successfully."""
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
    )


# ---------------------------------------------------------------------------
# GET /model/version
# ---------------------------------------------------------------------------
@app.get("/model/version", response_model=ModelVersionResponse)
async def model_version() -> ModelVersionResponse:
    """Return model version and training metrics."""
    if _metrics is None:
        raise HTTPException(
            status_code=503,
            detail="Model metrics are not available. Service is degraded.",
        )

    return ModelVersionResponse(
        model_version=_MODEL_VERSION,
        trained_at=_metrics.get("timestamp", "unknown"),
        metrics={
            "accuracy": _metrics.get("accuracy"),
            "precision_macro": _metrics.get("precision_macro"),
            "recall_macro": _metrics.get("recall_macro"),
            "f1_macro": _metrics.get("f1_macro"),
            "roc_auc": _metrics.get("roc_auc"),
        },
    )
