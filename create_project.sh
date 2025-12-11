#!/bin/bash
# This script will create the complete AI DDoS Shield project structure and files.

echo "🚀 Starting project creation..."

# --- Create Directories ---
mkdir -p frontend
mkdir -p simulator

# --- Create Main Application Files ---

# run.sh
cat <<'EOF' > run.sh
#!/usr/bin/env bash
set -euo pipefail

# Ensure the script is run with sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or with sudo"
  exit 1
fi

echo "Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing/updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting AI DDoS Shield on http://0.0.0.0:8000"
# The --reload flag is useful for development but can be removed for production
exec uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info --reload
EOF

# requirements.txt
cat <<'EOF' > requirements.txt
fastapi>=0.95
uvicorn[standard]>=0.22
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
joblib>=1.3
python-multipart>=0.0.6
websockets>=11.0
aiofiles>=23.1
EOF

# config.py
cat <<'EOF' > config.py
import os

# --- General Settings ---
WINDOW_SECONDS = 60  # Time window for tracking requests in seconds
ATTACK_BLOCK_SECONDS = 300  # Default duration to block an IP after detection

# --- ML Model & Data Paths ---
DATASET_DIR = "dataset"
MODEL_DIR = "models"
GENERATED_DATASET_PATH = os.path.join(DATASET_DIR, "generated_traffic.csv")
SUPERVISED_MODEL_PATH = os.path.join(MODEL_DIR, "ddos_supervised.pkl")
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "ddos_anom.pkl")
SIGNATURES_PATH = "signatures.txt"

# --- Detection Thresholds ---
# The model score (0.0 to 1.0) above which an IP is considered malicious.
DETECTION_THRESHOLD = 0.9

# --- Rate Limiter Settings ---
DEFAULT_RATE_LIMIT = 50.0  # Allowed requests per second for a single IP
DEFAULT_BURST_LIMIT = 200    # Burst capacity for a single IP
EOF

# generate_dataset.py
cat <<'EOF' > generate_dataset.py
import pandas as pd
import numpy as np
import os
import config

def generate_traffic_data(n_samples=50000):
    """
    Generates a synthetic dataset of HTTP traffic features for training.
    Features are designed to be extractable from a web server context.
    """
    print(f"Generating a synthetic dataset with {n_samples} samples...")
    
    # --- Normal Traffic ---
    n_normal = int(n_samples * 0.95)
    normal_data = {
        'req_rate': np.random.uniform(0.1, 10, n_normal), # Requests per second from this IP
        'unique_paths_rate': np.random.uniform(0.1, 5, n_normal), # Unique URLs per second
        'ip_entropy': np.random.uniform(0.5, 4.0, n_normal), # A measure of randomness in recent source IPs
        'payload_size': np.random.randint(50, 2048, n_normal),
        'label': 0 # 0 for 'normal'
    }
    normal_df = pd.DataFrame(normal_data)

    # --- DDoS Traffic ---
    n_ddos = n_samples - n_normal
    ddos_data = {
        'req_rate': np.random.uniform(50, 500, n_ddos), # Very high request rate
        'unique_paths_rate': np.random.uniform(0.01, 1, n_ddos), # Often hit the same URL
        'ip_entropy': np.random.uniform(0.0, 0.5, n_ddos), # Low entropy if it's a single attacker
        'payload_size': np.random.randint(10, 500, n_ddos), # Often smaller, simpler requests
        'label': 1 # 1 for 'ddos'
    }
    ddos_df = pd.DataFrame(ddos_data)

    # Combine and shuffle
    df = pd.concat([normal_df, ddos_df]).sample(frac=1).reset_index(drop=True)
    
    # Ensure directory exists
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    
    # Save to CSV
    df.to_csv(config.GENERATED_DATASET_PATH, index=False)
    print(f"Successfully generated and saved dataset to '{config.GENERATED_DATASET_PATH}'")
    print("\nDataset preview:")
    print(df.head())
    print("\nClass distribution:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    generate_traffic_data()
EOF

# train.py
cat <<'EOF' > train.py
import os
import pandas as pd
import argparse
import config
from detector import HybridDetector

def main():
    if not os.path.exists(config.GENERATED_DATASET_PATH):
        print(f"Error: Dataset not found at '{config.GENERATED_DATASET_PATH}'.")
        print("Please run 'python3 generate_dataset.py' first to create it.")
        return

    print("Initializing detector and starting training process...")
    
    # Ensure model directory exists
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    hd = HybridDetector()
    
    print("\n--- Training Supervised Model ---")
    hd.train_supervised(config.GENERATED_DATASET_PATH)
    
    print("\n--- Training Anomaly Model ---")
    hd.train_anomaly(config.GENERATED_DATASET_PATH)
    
    print("\n✅ Training finished successfully.")
    print(f"Models saved to '{config.SUPERVISED_MODEL_PATH}' and '{config.ANOMALY_MODEL_PATH}'.")

if __name__ == "__main__":
    main()
EOF

# detector.py
cat <<'EOF' > detector.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Optional, Tuple, List, Dict, Any
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
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

def load_signatures(path: str = config.SIGNATURES_PATH) -> List[str]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s.lower())
    return out

class HybridDetector:
    def __init__(self):
        self.supervised: Optional[RandomForestClassifier] = None
        self.anomaly: Optional[IsolationForest] = None
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

    def train_anomaly(self, csv_path: str):
        X, y = load_generated_csv(csv_path)
        if X is None:
            logging.warning("No data for anomaly training, skipping.")
            return
            
        # We can train the anomaly detector on all data or just normal data.
        # Training on just normal data often yields better results.
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
            
        self.signatures = load_signatures()
        logging.info(f"Loaded {len(self.signatures)} signatures.")

    def signature_check(self, payload: str) -> bool:
        if not payload:
            return False
        p = payload.lower()
        for s in self.signatures:
            if s in p:
                return True
        return False

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
        # decision_function returns negative for anomalies, positive for inliers.
        # We normalize it to a 0-1 score where higher is more anomalous.
        score = self.anomaly.score_samples(arr)[0]
        # A simple normalization. This can be improved with calibration.
        normalized_score = 1 - max(0, min(1, (score + 0.5) * 2))
        return float(normalized_score)

    def hybrid_decision(self, features: List[float], payload: str | None = None) -> Dict[str, Any]:
        payload_str = payload or ""
        is_signature_match = self.signature_check(payload_str)
        
        if is_signature_match:
            return {"decision": True, "reason": "signature", "score": 1.0, "details": "Payload matched a known malicious signature."}

        if not self.supervised or not self.anomaly:
            return {"decision": False, "reason": "normal", "score": 0.0, "details": "ML models not loaded."}
            
        supervised_prob = self.supervised_proba(features)
        anomaly_val = self.anomaly_score(features)
        
        # Combine scores: take the higher of the two probabilities
        final_score = max(supervised_prob, anomaly_val)

        if final_score >= config.DETECTION_THRESHOLD:
            reason = "anomaly" if anomaly_val > supervised_prob else "supervised"
            return {"decision": True, "reason": f"ml_{reason}", "score": final_score, "details": f"Supervised Prob: {supervised_prob:.2f}, Anomaly Score: {anomaly_val:.2f}"}

        return {"decision": False, "reason": "normal", "score": final_score, "details": f"Supervised Prob: {supervised_prob:.2f}, Anomaly Score: {anomaly_val:.2f}"}
EOF

# rate_limiter.py
cat <<'EOF' > rate_limiter.py
import time
import threading
from typing import Dict
import config

class TokenBucket:
    def __init__(self, rate: float, burst: int):
        self.rate = float(rate)
        self.capacity = int(burst)
        self._tokens = self.capacity
        self._last_refill_time = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill_time
            self._last_refill_time = now

            # Refill tokens
            self._tokens += elapsed * self.rate
            self._tokens = min(self.capacity, self._tokens)

            # Check if there are enough tokens and consume
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

class RateLimiterManager:
    def __init__(self, default_rate=config.DEFAULT_RATE_LIMIT, default_burst=config.DEFAULT_BURST_LIMIT):
        self.buckets: Dict[str, TokenBucket] = {}
        self.default_rate = default_rate
        self.default_burst = default_burst
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if key not in self.buckets:
            with self.lock:
                # Double-check in case another thread created it
                if key not in self.buckets:
                    self.buckets[key] = TokenBucket(self.default_rate, self.default_burst)
        
        return self.buckets[key].consume(1)

    def set_rate(self, key: str, rate: float, burst: int):
        with self.lock:
            self.buckets[key] = TokenBucket(rate, burst)
EOF

# firewall.py
cat <<'EOF' > firewall.py
import subprocess
import logging
import time
from typing import Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FirewallManager:
    def __init__(self):
        # Use a set for fast lookups of currently blocked IPs
        self.blocked_ips: Set[str] = set()
        self._sync_from_iptables()

    def _run_command(self, command: list[str]) -> bool:
        """Runs a shell command, logs output, and returns success status."""
        try:
            # We must use 'sudo' to modify iptables
            full_command = ['sudo'] + command
            process = subprocess.run(
                full_command,
                check=True,
                capture_output=True,
                text=True
            )
            logging.info(f"Successfully executed: {' '.join(full_command)}")
            if process.stdout:
                logging.debug(f"stdout: {process.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing: {' '.join(e.cmd)}")
            logging.error(f"Return code: {e.returncode}")
            logging.error(f"Stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            logging.error("Error: 'sudo' or 'iptables' command not found. Is it in the system's PATH?")
            return False

    def _sync_from_iptables(self):
        """Syncs the internal state with the actual iptables rules."""
        logging.info("Syncing blocked IPs from iptables rules...")
        self.blocked_ips.clear()
        try:
            # List all rules in the INPUT chain
            result = subprocess.run(
                ['sudo', 'iptables', '-L', 'INPUT', '-n'],
                capture_output=True, text=True, check=True
            )
            for line in result.stdout.splitlines():
                # Look for lines that indicate a DROP rule for a specific source IP
                if 'DROP' in line and '0.0.0.0/0' not in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        ip = parts[3]
                        self.blocked_ips.add(ip)
            logging.info(f"Sync complete. Found {len(self.blocked_ips)} blocked IPs.")
        except Exception as e:
            logging.error(f"Could not sync from iptables: {e}")

    def block_ip(self, ip: str) -> bool:
        """Blocks a given IP address using iptables."""
        if ip in self.blocked_ips:
            logging.warning(f"IP {ip} is already blocked.")
            return True
        
        # We insert the rule at the top of the INPUT chain
        command = ['iptables', '-I', 'INPUT', '1', '-s', ip, '-j', 'DROP']
        success = self._run_command(command)
        if success:
            self.blocked_ips.add(ip)
            logging.info(f"Successfully blocked IP: {ip}")
        return success

    def unblock_ip(self, ip: str) -> bool:
        """Unblocks a given IP address by deleting the corresponding iptables rule."""
        if ip not in self.blocked_ips:
            logging.warning(f"IP {ip} is not in the active block list.")
            return True
            
        command = ['iptables', '-D', 'INPUT', '-s', ip, '-j', 'DROP']
        success = self._run_command(command)
        if success:
            if ip in self.blocked_ips:
                self.blocked_ips.remove(ip)
            logging.info(f"Successfully unblocked IP: {ip}")
        return success

    def is_blocked(self, ip: str) -> bool:
        """Checks if an IP is in the set of blocked IPs."""
        return ip in self.blocked_ips

# Create a global instance to be used by the app
firewall = FirewallManager()
EOF

# app.py
cat <<'EOF' > app.py
import time
import threading
import asyncio
import json
import logging
from collections import defaultdict, deque
from typing import Dict, Deque, Any, List, Set

from fastapi import FastAPI, Request, Response, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import config
from detector import HybridDetector
from rate_limiter import RateLimiterManager
from firewall import firewall

# --- App Initialization ---
app = FastAPI(title="AI DDoS Hybrid Shield")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the frontend directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# --- Global State & Models ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In-memory data stores for real-time metrics
lock = threading.RLock()
per_ip_requests: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=2000))
per_ip_paths: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=500))
global_requests: Deque[float] = deque(maxlen=20000)
event_log: Deque[Dict] = deque(maxlen=200)

# Load models and managers
detector = HybridDetector()
rate_mgr = RateLimiterManager()
_block_expirations: Dict[str, float] = {} # Tracks when to unblock an IP

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

ws_manager = ConnectionManager()


# --- Background Tasks ---
async def unblock_expired_ips():
    """Periodically checks for and unblocks IPs whose block duration has expired."""
    while True:
        await asyncio.sleep(5)
        now = time.time()
        with lock:
            expired_ips = [ip for ip, expiry in _block_expirations.items() if now >= expiry]
        
        if expired_ips:
            logger.info(f"Found expired IPs to unblock: {expired_ips}")
            for ip in expired_ips:
                if firewall.unblock_ip(ip):
                    with lock:
                        _block_expirations.pop(ip, None)
                    log_event("auto_unblock", ip, "Block duration expired")

async def broadcast_metrics():
    """Periodically computes and broadcasts metrics to all connected WebSocket clients."""
    while True:
        await asyncio.sleep(2)
        if ws_manager.active_connections:
            metrics = get_metrics_data()
            await ws_manager.broadcast(json.dumps(metrics))

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up AI DDoS Shield...")
    detector.load_models()
    # Start background tasks
    asyncio.create_task(unblock_expired_ips())
    asyncio.create_task(broadcast_metrics())
    logger.info("Application startup complete.")


# --- Helper Functions ---
def get_client_ip(request: Request) -> str:
    """Extracts the client's IP address from the request."""
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def log_event(kind: str, ip: str, reason: str, details: Any = None):
    """Adds a structured event to the event log."""
    event = {"ts": time.time(), "kind": kind, "ip": ip, "reason": reason, "details": details or {}}
    event_log.appendleft(event)
    logger.info(f"EVENT: {kind} - IP: {ip} - Reason: {reason}")

def block_ip_action(ip: str, reason: str, details: Dict):
    """Handles the logic for blocking an IP."""
    if not firewall.is_blocked(ip):
        if firewall.block_ip(ip):
            with lock:
                _block_expirations[ip] = time.time() + config.ATTACK_BLOCK_SECONDS
            log_event("auto_block", ip, reason, details)
        else:
            log_event("block_failed", ip, "iptables command failed", details)
    else:
        # If already blocked, just extend the expiration
        with lock:
            _block_expirations[ip] = time.time() + config.ATTACK_BLOCK_SECONDS
        logger.warning(f"IP {ip} already blocked. Extending block time.")


# --- Core Middleware ---
@app.middleware("http")
async def threat_detection_middleware(request: Request, call_next):
    ip = get_client_ip(request)
    now = time.monotonic()
    
    # 1. Check Firewall First (most efficient)
    if firewall.is_blocked(ip):
        return Response(content="Blocked by firewall", status_code=403)

    # 2. Check Rate Limiter
    if not rate_mgr.allow(ip):
        log_event("rate_limit", ip, "Rate limit exceeded")
        return Response(content="Rate limit exceeded", status_code=429)

    # 3. Record request for analysis
    with lock:
        global_requests.append(now)
        per_ip_requests[ip].append(now)
        per_ip_paths[ip].append(request.url.path)
        
        # Prune old entries
        while global_requests and global_requests[0] < now - config.WINDOW_SECONDS:
            global_requests.popleft()
        while per_ip_requests[ip] and per_ip_requests[ip][0] < now - config.WINDOW_SECONDS:
            per_ip_requests[ip].popleft()

    # 4. Feature Extraction
    req_rate = len(per_ip_requests[ip]) / config.WINDOW_SECONDS
    unique_paths_rate = len(set(per_ip_paths[ip])) / config.WINDOW_SECONDS
    ip_entropy = 0.0 # Placeholder for a more complex entropy calculation
    payload_size = int(request.headers.get('content-length', 0))
    
    features = [req_rate, unique_paths_rate, ip_entropy, payload_size]
    
    # 5. Hybrid Detection
    payload_preview = ""
    try:
        payload_preview = (await request.body())[:1024].decode("utf-8", errors="ignore")
    except Exception:
        pass
        
    result = detector.hybrid_decision(features, payload_preview)
    
    if result["decision"]:
        block_ip_action(ip, result["reason"], result)
        return Response(content="Threat detected and IP blocked", status_code=403)

    response = await call_next(request)
    return response


# --- API Endpoints ---
@app.get("/", include_in_schema=False)
async def read_index():
    return FileResponse('frontend/index.html')

def get_metrics_data() -> Dict:
    """Computes the current state of metrics for the API and websockets."""
    now = time.monotonic()
    active_ips_data = []
    
    with lock:
        active_ips = list(per_ip_requests.keys())
        for ip in active_ips:
            # Clean up inactive IPs to prevent memory leaks
            if not per_ip_requests[ip] or now - per_ip_requests[ip][-1] > config.WINDOW_SECONDS * 2:
                del per_ip_requests[ip]
                del per_ip_paths[ip]
                continue
            
            rate = len(per_ip_requests[ip]) / config.WINDOW_SECONDS
            active_ips_data.append({"ip": ip, "rate": round(rate, 2), "count": len(per_ip_requests[ip])})
            
        blocked_list = [{"ip": k, "expires": int(v - time.time())} for k, v in _block_expirations.items() if v > time.time()]
        event_list = list(event_log)

    return {
        "global_rate": round(len(global_requests) / config.WINDOW_SECONDS, 2),
        "active_ips_count": len(active_ips_data),
        "blocked_ips_count": len(blocked_list),
        "top_active_ips": sorted(active_ips_data, key=lambda x: x["rate"], reverse=True)[:10],
        "blocked_ips": blocked_list,
        "events": event_list,
    }

@app.get("/metrics")
async def metrics():
    return get_metrics_data()

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

@app.post("/block", status_code=200)
async def manual_block(req: Dict[str, Any]):
    ip = req.get("ip")
    seconds = int(req.get("seconds", config.ATTACK_BLOCK_SECONDS))
    if not ip:
        raise HTTPException(status_code=400, detail="IP address is required")
    block_ip_action(ip, "manual_block", {"source": "api"})
    with lock:
        _block_expirations[ip] = time.time() + seconds
    return {"message": f"IP {ip} blocked successfully."}

@app.post("/unblock", status_code=200)
async def manual_unblock(req: Dict[str, Any]):
    ip = req.get("ip")
    if not ip:
        raise HTTPException(status_code=400, detail="IP address is required")
    
    if firewall.unblock_ip(ip):
        with lock:
            _block_expirations.pop(ip, None)
        log_event("manual_unblock", ip, "Unblocked via API")
        return {"message": f"IP {ip} unblocked successfully."}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to unblock IP {ip}. Check logs.")

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}
EOF

# signatures.txt
cat <<'EOF' > signatures.txt
# Common web attack and scanning patterns
/etc/passwd
/bin/sh
wget http
curl http
union select
drop table
' or 1=1--
<script>
nmap
zgrab
masscan
EOF

# README.md
cat <<'EOF' > README.md
# AI-Powered DDoS Hybrid Shield

This project is a FastAPI application designed to detect and prevent DDoS attacks using a hybrid approach that combines machine learning, signature analysis, rate limiting, and a real firewall (`iptables`) for blocking malicious IP addresses.

## Features

-   **Hybrid Detection:** Uses a combination of a Random Forest classifier, an Isolation Forest for anomaly detection, and substring-based signature matching.
-   **Real-time Firewalling:** Directly integrates with Linux `iptables` to block and unblock IP addresses at the kernel level.
-   **Rate Limiting:** A token-bucket algorithm limits the number of requests per IP.
-   **Live Dashboard:** A real-time frontend built with vanilla JS and Chart.js that visualizes traffic, blocked IPs, and security events via WebSockets.
-   **Relevant ML Model:** Includes a script to generate a synthetic dataset based on realistic HTTP traffic patterns, ensuring the models are trained on features the application can actually observe.

## How It Works

1.  **Middleware:** Every incoming request is intercepted by a FastAPI middleware.
2.  **Filtering:** The middleware first checks if the source IP is already blocked by `iptables` or is exceeding the rate limit.
3.  **Feature Extraction:** For allowed requests, it calculates features in real-time (e.g., request rate from the IP, number of unique paths visited).
4.  **Hybrid Analysis:** These features are fed into the hybrid detector:
    -   The ML models (Supervised and Anomaly) calculate a threat score.
    -   The request payload is scanned for malicious signatures.
5.  **Blocking:** If the detector flags the request as malicious, the source IP is blocked using `iptables` for a configured duration, and a security event is logged.
6.  **Live Updates:** A background task pushes metrics and event logs to all connected dashboard clients via WebSockets.

## Setup and Installation

### 1. Prerequisites

-   A Linux environment (tested on Kali Linux/Debian).
-   Python 3.8+ and `pip`.
-   Root or `sudo` access.

### 2. Sudo Configuration (CRITICAL)

The application needs permission to run `iptables` without a password prompt.

1.  Open the sudoers file: `sudo visudo`
2.  Add this line at the end, replacing `your_user` with your username:
    ```
    your_user ALL=(ALL) NOPASSWD: /usr/sbin/iptables
    ```
3.  Save the file. This allows your user to execute `iptables` with `sudo` without being asked for a password.

### 3. Application Setup

1.  **Clone the repository or create the files from the provided code.**

2.  **Make the run script executable:**
    ```bash
    chmod +x run.sh
    ```

3.  **Generate the Dataset:**
    The ML models need to be trained on data that reflects the features we can extract from HTTP traffic.
    ```bash
    python3 generate_dataset.py
    ```
    This will create `dataset/generated_traffic.csv`.

4.  **Train the Models:**
    ```bash
    python3 train.py
    ```
    This will train the models and save `ddos_supervised.pkl` and `ddos_anom.pkl`.

### 4. Running the Application

**Important:** You must use `sudo` to allow the application to interact with `iptables`.

```bash
sudo ./run.sh
