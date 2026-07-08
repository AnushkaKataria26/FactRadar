import logging
from pathlib import Path
import joblib

from lime.lime_text import LimeTextExplainer
from src.preprocessing.clean_text import clean_text

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("models/v0.1_baseline.joblib")

_pipeline = None
_explainer = None

try:
    _pipeline = joblib.load(_MODEL_PATH)
    logger.info("Model loaded successfully in explainer from %s", _MODEL_PATH)
    # The classes in the pipeline are typically [0, 1] for real, fake
    _explainer = LimeTextExplainer(class_names=["real", "fake"])
except Exception as exc:
    logger.error("Failed to load model or instantiate explainer: %s", exc)

def explain_instance(text: str, num_features: int = 10) -> list[tuple[str, float]]:
    """
    Explain a single instance using LIME.
    
    Args:
        text (str): Raw input text.
        num_features (int): Maximum number of features to include in explanation.
        
    Returns:
        A list of (token, weight) tuples sorted by absolute weight descending.
    """
    if _pipeline is None or _explainer is None:
        raise RuntimeError("Explainer or pipeline is not loaded.")
        
    cleaned_text = clean_text(text)
    
    if not cleaned_text.strip():
        raise ValueError("Input contains no usable text content after preprocessing.")
        
    # Generate explanation
    exp = _explainer.explain_instance(
        cleaned_text, 
        _pipeline.predict_proba, 
        num_features=num_features
    )
    
    # exp.as_list() returns a list of (word, weight) for the predicted class
    explanation_list = exp.as_list()
    
    # Sort by absolute weight descending
    explanation_list.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # LIME might return fewer than num_features if the document is very short, 
    # but we just return what it gives (up to num_features).
    return explanation_list
