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
MLP_MODEL_PATH = os.path.join(MODEL_DIR, "ddos_mlp.pkl")
AUTOENCODER_MODEL_PATH = os.path.join(MODEL_DIR, "ddos_autoenc.pkl")
XSS_MODEL_PATH = os.path.join("xss", "model_xss.pkl")
SQL_MODEL_PATH = os.path.join("sql_injection", "model_sql.pkl")
SIGNATURES_PATH = "signatures.txt"

# --- Detection Thresholds ---
# The model score (0.0 to 1.0) above which an IP is considered malicious.
DETECTION_THRESHOLD = 0.7
MLP_THRESHOLD = 0.65
AUTOENCODER_THRESHOLD = 0.7
XSS_THRESHOLD = 0.55
SQL_THRESHOLD = 0.55

# --- Rate Limiter Settings ---
DEFAULT_RATE_LIMIT = 50.0  # Allowed requests per second for a single IP
DEFAULT_BURST_LIMIT = 200    # Burst capacity for a single IP
# --- Whitelist Settings ---
# IPs in this list will never be blocked by the firewall.
IP_WHITELIST = [
    "127.0.0.1",  # Localhost for accessing the dashboard
    "::1"         # Localhost for IPv6
]
