"""
Advanced XSS Inference using Ensemble + Neural Network Combination.
Falls back to signature detection if models unavailable.
"""
import sys
import re
from pathlib import Path
import joblib
import numpy as np

# TensorFlow is optional; if unavailable or incompatible, we skip neural scoring.
try:
    import tensorflow as tf  # noqa: F401
except Exception:
    tf = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

# XSS Signature patterns
_SIG_PATTERNS = [
    r'<\s*script[^>]*>.*?<\s*/\s*script\s*>',
    r'on\w+\s*=',  # Event handlers
    r'javascript:',
    r'<\s*iframe[^>]*>',
    r'<\s*embed[^>]*>',
    r'<\s*object[^>]*>',
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _SIG_PATTERNS]

# Model cache
_ENSEMBLE_MODEL = None
_NEURAL_MODEL = None
_VECTORIZER = None
_SCALER = None

def _load_models():
    """Load all XSS detection models"""
    global _ENSEMBLE_MODEL, _NEURAL_MODEL, _VECTORIZER, _SCALER
    
    if _ENSEMBLE_MODEL is not None:
        return  # Already loaded
    
    model_dir = Path(config.XSS_MODEL_PATH).parent
    
    try:
        _VECTORIZER = joblib.load(model_dir / 'xss_vectorizer.pkl')
        _ENSEMBLE_MODEL = joblib.load(model_dir / 'xss_ensemble.pkl')
        # Scaler / neural are optional
        scaler_path = model_dir / 'xss_scaler.pkl'
        nn_path = model_dir / 'xss_neural_network.h5'
        if scaler_path.exists():
            _SCALER = joblib.load(scaler_path)
        if tf is not None and nn_path.exists():
            _NEURAL_MODEL = tf.keras.models.load_model(str(nn_path))
    except Exception as e:
        print(f"⚠️ Could not load advanced XSS models: {e}")

def predict(payload: str) -> dict:
    """
    Detect XSS attacks using signature + ensemble + neural network.
    
    Returns:
        {
            'decision': bool,
            'score': float (0-1),
            'reason': str,
            'signature_match': bool,
            'ensemble_score': float,
            'neural_score': float
        }
    """
    
    # Check signatures first
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(payload):
            return {
                'decision': True,
                'score': 0.99,
                'reason': 'XSS signature match',
                'signature_match': True,
                'ensemble_score': 0.0,
                'neural_score': 0.0
            }
    
    # Load models if not already loaded
    _load_models()
    
    ensemble_score = 0.0
    neural_score = 0.0
    
    # Ensemble prediction
    if _ENSEMBLE_MODEL and _VECTORIZER:
        try:
            X_tfidf = _VECTORIZER.transform([payload])
            ensemble_score = float(_ENSEMBLE_MODEL.predict_proba(X_tfidf)[0, 1])
        except Exception as e:
            print(f"⚠️ Ensemble prediction failed: {e}")
    
    # Neural network prediction
    if _NEURAL_MODEL and _VECTORIZER and _SCALER and tf is not None:
        try:
            X_tfidf = _VECTORIZER.transform([payload]).toarray()
            X_scaled = _SCALER.transform(X_tfidf)
            neural_score = float(_NEURAL_MODEL.predict(X_scaled, verbose=0)[0, 0])
        except Exception as e:
            print(f"⚠️ Neural network prediction failed: {e}")
    
    # Combine scores (average of both models)
    combined_score = (ensemble_score + neural_score) / 2 if (ensemble_score + neural_score) > 0 else 0
    threshold = config.XSS_THRESHOLD
    
    return {
        'decision': combined_score >= threshold,
        'score': combined_score,
        'reason': f'XSS ML detection (ensemble: {ensemble_score:.3f}, nn: {neural_score:.3f})',
        'signature_match': False,
        'ensemble_score': ensemble_score,
        'neural_score': neural_score
    }
