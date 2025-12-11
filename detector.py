from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Optional, Tuple, List, Dict, Any
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_generated_csv(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not os.path.exists(path):
        logging.error(f"Dataset not found at {path}")
        return None, None
    try:
        df = pd.read_csv(path)
        if "label" not in df.columns:
            logging.error("Dataset must contain a 'label' column.")
            return None, None
        
        y = df["label"].values
        X = df.drop(columns=["label"]).values
        return X, y
    except Exception as e:
        logging.error(f"Failed to load or parse CSV at {path}: {e}")
        return None, None

def load_signatures(path: str | None = None) -> List[str]:
    sig_path = path or config.SIGNATURES_PATH
    if not os.path.exists(sig_path):
        return []
    out = []
    with open(sig_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s.lower())
    return out

class HybridDetector:
    def __init__(self):
        self.supervised: Optional[RandomForestClassifier] = None
        self.anomaly: Optional[IsolationForest] = None
        self.mlp: Optional[Pipeline] = None
        self.autoenc: Optional[Pipeline] = None
        self.signatures = []
        self.feature_names: List[str] = []

    def train_supervised(self, csv_path: str):
        X, y = load_generated_csv(csv_path)
        if X is None:
            logging.warning("No data for supervised training, skipping.")
            return
        
        df = pd.read_csv(csv_path)
        self.feature_names = [col for col in df.columns if col != 'label']
        logging.info(f"Training supervised model with features: {self.feature_names}")

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight="balanced")
        clf.fit(Xtr, ytr)
        
        logging.info("Supervised model trained. Evaluation report:")
        print(classification_report(yte, clf.predict(Xte)))
        
        joblib.dump(clf, config.SUPERVISED_MODEL_PATH)
        self.supervised = clf
        logging.info(f"Supervised model saved to {config.SUPERVISED_MODEL_PATH}")

    def train_mlp(self, csv_path: str):
        X, y = load_generated_csv(csv_path)
        if X is None:
            logging.warning("No data for MLP training, skipping.")
            return

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
        mlp = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=30, random_state=42, verbose=False))
        ])
        mlp.fit(Xtr, ytr)

        logging.info("MLP model trained. Evaluation report:")
        print(classification_report(yte, mlp.predict(Xte)))

        joblib.dump(mlp, config.MLP_MODEL_PATH)
        self.mlp = mlp
        logging.info(f"MLP model saved to {config.MLP_MODEL_PATH}")

    def train_autoencoder(self, csv_path: str):
        X, y = load_generated_csv(csv_path)
        if X is None:
            logging.warning("No data for autoencoder training, skipping.")
            return

        normal_X = X[y == 0]
        if normal_X.size == 0:
            logging.warning("No normal traffic rows for autoencoder training, skipping.")
            return

        Xtr, Xte = train_test_split(normal_X, test_size=0.2, random_state=42)
        ae = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MLPRegressor(hidden_layer_sizes=(32, 16, 32), activation="relu", solver="adam", max_iter=40, random_state=42, verbose=False))
        ])
        ae.fit(Xtr, Xtr)

        # Compute simple reconstruction error stats for reference
        recon = ae.predict(Xte)
        mse = np.mean((recon - Xte) ** 2)
        logging.info(f"Autoencoder trained. Validation MSE: {mse:.4f}")

        joblib.dump(ae, config.AUTOENCODER_MODEL_PATH)
        self.autoenc = ae
        logging.info(f"Autoencoder model saved to {config.AUTOENCODER_MODEL_PATH}")

    def train_anomaly(self, csv_path: str):
        X, y = load_generated_csv(csv_path)
        if X is None:
            logging.warning("No data for anomaly training, skipping.")
            return
            
        normal_X = X[y == 0]
        iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        iso.fit(normal_X)
        
        joblib.dump(iso, config.ANOMALY_MODEL_PATH)
        self.anomaly = iso
        logging.info(f"Anomaly model saved to {config.ANOMALY_MODEL_PATH}")

    def load_models(self):
        if os.path.exists(config.SUPERVISED_MODEL_PATH):
            self.supervised = joblib.load(config.SUPERVISED_MODEL_PATH)
            logging.info("Loaded supervised model.")
        else:
            logging.warning("Supervised model not found.")
            
        if os.path.exists(config.ANOMALY_MODEL_PATH):
            self.anomaly = joblib.load(config.ANOMALY_MODEL_PATH)
            logging.info("Loaded anomaly model.")
        else:
            logging.warning("Anomaly model not found.")

        if os.path.exists(config.MLP_MODEL_PATH):
            try:
                self.mlp = joblib.load(config.MLP_MODEL_PATH)
                logging.info("Loaded MLP model.")
            except Exception as e:
                self.mlp = None
                logging.error(f"Failed to load MLP model, skipping: {e}")
        else:
            logging.warning("MLP model not found.")

        if os.path.exists(config.AUTOENCODER_MODEL_PATH):
            try:
                self.autoenc = joblib.load(config.AUTOENCODER_MODEL_PATH)
                logging.info("Loaded autoencoder model.")
            except Exception as e:
                self.autoenc = None
                logging.error(f"Failed to load autoencoder model, skipping: {e}")
        else:
            logging.warning("Autoencoder model not found.")
            
        self.signatures = load_signatures()
        logging.info(f"Loaded {len(self.signatures)} signatures.")

    def signature_check(self, payload: str) -> str | None:
        """Checks for a signature and returns the matched signature string, or None."""
        if not payload:
            return None
        p = payload.lower()
        for s in self.signatures:
            if s in p:
                return s  # Return the signature that was found
        return None

    def supervised_proba(self, features: List[float]) -> float:
        if self.supervised is None:
            return 0.0
        arr = np.array(features, dtype=float).reshape(1, -1)
        try:
            return float(self.supervised.predict_proba(arr)[0][1])
        except Exception:
            return 0.0

    def anomaly_score(self, features: List[float]) -> float:
        if self.anomaly is None:
            return 0.0
        arr = np.array(features, dtype=float).reshape(1, -1)
        score = self.anomaly.score_samples(arr)[0]
        normalized_score = 1 - max(0, min(1, (score + 0.5) * 2))
        return float(normalized_score)

    def mlp_proba(self, features: List[float]) -> float:
        if self.mlp is None:
            return 0.0
        arr = np.array(features, dtype=float).reshape(1, -1)
        try:
            proba = self.mlp.predict_proba(arr)[0][1]
            return float(proba)
        except Exception:
            return 0.0

    def autoencoder_score(self, features: List[float]) -> float:
        if self.autoenc is None:
            return 0.0
        arr = np.array(features, dtype=float).reshape(1, -1)
        try:
            recon = self.autoenc.predict(arr)
            mse = float(np.mean((recon - arr) ** 2))
            # Normalize reconstruction error to [0,1] via sigmoid-like mapping
            score = 1 - (1 / (1 + mse))
            return score
        except Exception:
            return 0.0

    def hybrid_decision(self, features: List[float], payload: str | None = None) -> Dict[str, Any]:
        payload_str = payload or ""
        matched_signature = self.signature_check(payload_str)
        
        if matched_signature:
            return {"decision": True, "reason": "signature", "score": 1.0, "details": f"Payload matched signature: '{matched_signature}'"}

        if not self.supervised and not self.anomaly and not self.mlp and not self.autoenc:
            return {"decision": False, "reason": "normal", "score": 0.0, "details": "ML models not loaded."}
            
        supervised_prob = self.supervised_proba(features)
        anomaly_val = self.anomaly_score(features)
        mlp_prob = self.mlp_proba(features)
        autoenc_val = self.autoencoder_score(features)

        final_score = max(supervised_prob, anomaly_val, mlp_prob, autoenc_val)
        threshold = max(config.DETECTION_THRESHOLD, config.MLP_THRESHOLD, config.AUTOENCODER_THRESHOLD)

        if final_score >= threshold:
            if final_score == autoenc_val:
                reason = "autoencoder"
            elif final_score == mlp_prob:
                reason = "mlp"
            elif final_score == anomaly_val:
                reason = "anomaly"
            else:
                reason = "supervised"
            return {
                "decision": True,
                "reason": f"ml_{reason}",
                "score": final_score,
                "details": f"Supervised: {supervised_prob:.2f}, Anomaly: {anomaly_val:.2f}, MLP: {mlp_prob:.2f}, AE: {autoenc_val:.2f}"
            }

        return {
            "decision": False,
            "reason": "normal",
            "score": final_score,
            "details": f"Supervised: {supervised_prob:.2f}, Anomaly: {anomaly_val:.2f}, MLP: {mlp_prob:.2f}, AE: {autoenc_val:.2f}"
        }
