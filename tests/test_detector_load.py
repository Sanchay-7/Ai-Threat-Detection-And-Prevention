import os
import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path

# Ensure project root is importable when running tests directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from detector import HybridDetector


def _write_dummy_models(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)
    X = np.random.rand(50, 4)
    y = np.random.randint(0, 2, size=50)

    rf = RandomForestClassifier(n_estimators=5, random_state=0)
    rf.fit(X, y)
    joblib.dump(rf, base_dir / "rf.pkl")

    iso = IsolationForest(n_estimators=5, random_state=0)
    iso.fit(X)
    joblib.dump(iso, base_dir / "iso.pkl")

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(8,), max_iter=10, random_state=0))
    ])
    mlp.fit(X, y)
    joblib.dump(mlp, base_dir / "mlp.pkl")

    autoenc = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", MLPRegressor(hidden_layer_sizes=(8,), max_iter=10, random_state=0))
    ])
    autoenc.fit(X, X)
    joblib.dump(autoenc, base_dir / "ae.pkl")


def test_load_models_with_all_components(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    _write_dummy_models(model_dir)

    # Point config paths to temp models and signature
    monkeypatch.setattr(config, "SUPERVISED_MODEL_PATH", str(model_dir / "rf.pkl"))
    monkeypatch.setattr(config, "ANOMALY_MODEL_PATH", str(model_dir / "iso.pkl"))
    monkeypatch.setattr(config, "MLP_MODEL_PATH", str(model_dir / "mlp.pkl"))
    monkeypatch.setattr(config, "AUTOENCODER_MODEL_PATH", str(model_dir / "ae.pkl"))

    sig_path = tmp_path / "sigs.txt"
    sig_path.write_text("malicious", encoding="utf-8")
    monkeypatch.setattr(config, "SIGNATURES_PATH", str(sig_path))

    hd = HybridDetector()
    hd.load_models()

    assert hd.supervised is not None
    assert hd.anomaly is not None
    assert hd.mlp is not None
    assert hd.autoenc is not None
    assert hd.signatures == ["malicious"]

    # Verify decision uses loaded models without raising
    features = [0.1, 0.1, 0.1, 10]
    decision = hd.hybrid_decision(features, payload=None)
    assert "decision" in decision and "score" in decision


def test_signature_match_short_circuits(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    _write_dummy_models(model_dir)

    monkeypatch.setattr(config, "SUPERVISED_MODEL_PATH", str(model_dir / "rf.pkl"))
    monkeypatch.setattr(config, "ANOMALY_MODEL_PATH", str(model_dir / "iso.pkl"))
    monkeypatch.setattr(config, "MLP_MODEL_PATH", str(model_dir / "mlp.pkl"))
    monkeypatch.setattr(config, "AUTOENCODER_MODEL_PATH", str(model_dir / "ae.pkl"))

    sig_path = tmp_path / "sigs.txt"
    sig_path.write_text("blockme", encoding="utf-8")
    monkeypatch.setattr(config, "SIGNATURES_PATH", str(sig_path))

    hd = HybridDetector()
    hd.load_models()

    decision = hd.hybrid_decision([0.0, 0.0, 0.0, 0], payload="please blockme now")
    assert decision["decision"] is True
    assert decision["reason"] == "signature"
