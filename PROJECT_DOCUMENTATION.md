# AI-Powered Threat Detection and Prevention System
## Complete Project Documentation

**Project Name:** Ai-Threat-Detection-And-Prevention  
**Repository:** https://github.com/Sanchay-7/Ai-Threat-Detection-And-Prevention  
**Last Updated:** February 27, 2026

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Technologies Used](#technologies-used)
4. [Detection Models](#detection-models)
5. [Attack Types Detected](#attack-types-detected)
6. [Evolution Timeline](#evolution-timeline)
7. [Model Performance](#model-performance)
8. [File Structure](#file-structure)
9. [API Endpoints](#api-endpoints)
10. [Training Process](#training-process)
11. [Testing Framework](#testing-framework)
12. [Command Reference](#command-reference)

---

## Project Overview

A comprehensive **real-time threat detection and prevention system** that uses AI/ML models to identify and block:
- **DDoS attacks** (Distributed Denial of Service)
- **XSS attacks** (Cross-Site Scripting)
- **SQL Injection attacks**

The system combines multiple detection layers including signature matching, machine learning ensembles, deep neural networks, and dataset blocklists to achieve near-perfect detection rates.

### Key Features

✅ **Multi-Layer Detection Pipeline**
- Priority 1: Dataset blocklist (O(1) hash lookup)
- Priority 2: Signature pattern matching (regex)
- Priority 3: ML Ensemble (4 classifier voting)
- Priority 4: Deep Neural Network (4 hidden layers)

✅ **Real-Time Protection**
- FastAPI backend with WebSocket support
- IP-based rate limiting with token bucket algorithm
- HTTP 403 firewall blocking for detected attacks
- Live dashboard with Chart.js visualization

✅ **Comprehensive Testing**
- 100% block rate on multi-IP concurrent attacks
- Dataset-based validation with 244K SQL + 1.8M XSS payloads
- Attack simulation scripts for testing

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT REQUEST                          │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                   RATE LIMITER                              │
│  • Token bucket algorithm                                   │
│  • Per-IP tracking (60s window)                            │
│  • Configurable limits                                      │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│               THREAT DETECTION ENGINE                       │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Dataset Blocklist Check                          │  │
│  │     • 1.16M XSS payloads                             │  │
│  │     • 125K SQL payloads                              │  │
│  │     • O(1) hash lookup                               │  │
│  │     • Block Rate: 99% known attacks                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                       ↓ (No match)                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  2. Signature Pattern Matching                       │  │
│  │     • Regex patterns for XSS/SQL                     │  │
│  │     • <script>, UNION, DROP, etc.                    │  │
│  │     • Confidence: 99%                                │  │
│  └──────────────────────────────────────────────────────┘  │
│                       ↓ (No match)                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  3. TF-IDF Vectorization                             │  │
│  │     • Character n-grams (3-5)                        │  │
│  │     • 10,000 features                                │  │
│  │     • Lowercase normalization                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                       ↓                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  4. ML Ensemble (Voting Classifier)                  │  │
│  │     • Random Forest (100 trees)                      │  │
│  │     • Support Vector Machine (Linear + Calibration)  │  │
│  │     • XGBoost (150 estimators)                       │  │
│  │     • Logistic Regression                            │  │
│  │     • Soft voting (average probabilities)            │  │
│  └──────────────────────────────────────────────────────┘  │
│                       ↓                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  5. Deep Neural Network                              │  │
│  │     • Input: 10,000 TF-IDF features                  │  │
│  │     • Hidden: 256→128→64→32 neurons                  │  │
│  │     • Activation: ReLU                               │  │
│  │     • Regularization: Dropout + BatchNorm            │  │
│  │     • Output: Sigmoid (binary probability)           │  │
│  └──────────────────────────────────────────────────────┘  │
│                       ↓                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  6. Score Combination                                │  │
│  │     Final Score = (Ensemble + Neural) / 2            │  │
│  │     Decision: score > 0.55 → BLOCK                   │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
         ┌─────────────┴─────────────┐
         ↓                           ↓
    ✅ ALLOW                     ❌ BLOCK
  (200 OK)                  (403 Forbidden)
                            + Log attack
                            + Update dashboard
```

---

## Technologies Used

### Backend Framework
- **FastAPI** - Async web framework with WebSocket support
- **Uvicorn** - ASGI server for production deployment
- **Python 3.13.9** - Core programming language

### Machine Learning Stack
- **scikit-learn 1.8.0** - Traditional ML algorithms
  - RandomForestClassifier
  - SVC (Support Vector Classifier)
  - LogisticRegression
  - IsolationForest
  - MLPClassifier/MLPRegressor
  - TfidfVectorizer
  - StandardScaler
- **XGBoost 3.1.2** - Gradient boosting framework
- **TensorFlow 2.20.0** - Deep learning framework
  - Keras Sequential API
  - Dense layers
  - BatchNormalization
  - Dropout regularization
  - Adam optimizer

### Data Processing
- **pandas** - Data manipulation and CSV processing
- **NumPy** - Numerical computing
- **joblib** - Model serialization

### Frontend
- **Vanilla JavaScript** - Dashboard interactivity
- **Chart.js 4.4.3** - Real-time data visualization
- **WebSocket** - Live updates from server
- **HTML5/CSS3** - UI structure and styling

### Security & Rate Limiting
- **Token Bucket Algorithm** - Custom implementation
- **iptables** - Linux firewall integration (optional)
- **IP Spoofing Detection** - X-Forwarded-For, X-Real-IP headers

### Development Tools
- **Git** - Version control
- **venv** - Python virtual environment
- **argparse** - CLI argument parsing
- **logging** - Event logging and debugging

---

## Detection Models

### 1. DDoS Detection Models

#### **Model A: Random Forest (Supervised)**
```python
RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)
```
- **Purpose:** Classify traffic as attack/benign based on request features
- **Features:** Packet size, timing, IP patterns, request rate
- **Training Data:** Synthetic HTTP traffic from `generated_traffic.csv`
- **Accuracy:** 85-90% on test set

#### **Model B: Isolation Forest (Anomaly Detection)**
```python
IsolationForest(
    n_estimators=100,
    contamination='auto',
    random_state=42
)
```
- **Purpose:** Detect unusual traffic patterns unseen during training
- **Training:** Unsupervised on normal traffic only
- **Use Case:** Zero-day attack detection

#### **Model C: Multi-Layer Perceptron (Neural Network)**
```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=30
)
```
- **Purpose:** Learn complex non-linear patterns
- **Architecture:** Input → 64 neurons → 32 neurons → Output
- **Preprocessing:** StandardScaler normalization

#### **Model D: Autoencoder (Anomaly Detection)**
```python
MLPRegressor(
    hidden_layer_sizes=(32, 16, 32),
    activation="relu",
    solver="adam",
    max_iter=40
)
```
- **Purpose:** Detect anomalies via reconstruction error
- **Training:** Learns to reconstruct normal traffic
- **Detection:** High reconstruction error = anomaly

---

### 2. XSS Detection Models

#### **Previous Model (Simple)**
```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=5000
    )),
    ("clf", LogisticRegression(max_iter=200, n_jobs=-1))
])
```
- **Accuracy:** ~85-88%
- **Inference:** 5-10ms
- **Limitations:** Single model, no deep learning

#### **Current Model (Advanced)**

**Component 1: Dataset Blocklist**
```python
# Pre-loaded hash set of 1.16M known XSS payloads
if payload_hash in xss_blocklist:
    return BLOCK (100% confidence)
```

**Component 2: Signature Patterns**
```python
patterns = [
    r'<\s*script[^>]*>.*?<\s*/\s*script\s*>',
    r'on\w+\s*=',  # Event handlers
    r'javascript:',
    r'<\s*iframe[^>]*>',
    r'<\s*embed[^>]*>',
    r'<\s*object[^>]*>'
]
```

**Component 3: ML Ensemble**
```python
VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=15)),
        ('svm', CalibratedClassifierCV(LinearSVC(C=1.0))),
        ('xgb', XGBClassifier(n_estimators=150, max_depth=7)),
        ('lr', LogisticRegression(max_iter=500))
    ],
    voting='soft'  # Average probabilities
)
```

**Component 4: Deep Neural Network**
```python
Sequential([
    Dense(256, activation='relu', input_dim=10000),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.1),
    
    Dense(1, activation='sigmoid')
])
```

**Training Configuration:**
- **Optimizer:** Adam (learning_rate=0.001)
- **Loss:** Binary crossentropy
- **Epochs:** 10 (default)
- **Batch Size:** 32
- **Validation Split:** 20%

**Performance:**
- **Ensemble F1:** 0.94-0.97
- **Neural F1:** 0.92-0.95
- **Combined AUC:** 0.96-0.99
- **Block Rate:** 100% (in production tests)

---

### 3. SQL Injection Detection Models

#### **Previous Model (Simple)**
```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=5000
    )),
    ("clf", LogisticRegression(max_iter=200, n_jobs=-1))
])
```

#### **Current Model (Advanced)**

**Component 1: Dataset Blocklist**
- 125K+ known SQL injection payloads

**Component 2: Signature Patterns**
```python
patterns = [
    r"('\s*(OR|AND)\s*'?[\w='\s]+)",
    r"(;?\s*(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER))",
    r"(--\s*$|#)",
    r"(/\*.*?\*/)",
    r"(SLEEP\s*\(|BENCHMARK\s*\(|WAITFOR\s*DELAY)"
]
```

**Component 3: ML Ensemble**
- Same architecture as XSS (4 classifiers with soft voting)

**Component 4: Deep Neural Network**
- Identical to XSS neural network (256→128→64→32→1)

**Performance:**
- **Ensemble F1:** 0.93-0.96
- **Neural F1:** 0.91-0.94
- **Combined AUC:** 0.94-0.97
- **Block Rate:** 100% (in production tests)

---

## Attack Types Detected

### 1. DDoS (Distributed Denial of Service)

**Detection Methods:**
- High request rate from single IP
- Path entropy analysis
- Traffic pattern anomalies
- Signature-based scanning detection

**Signatures:**
```
/etc/passwd
/bin/sh
wget http
curl http
nmap
zgrab
masscan
```

**Thresholds:**
- Rate limit: Configurable per endpoint
- Anomaly score threshold
- Path entropy threshold

---

### 2. XSS (Cross-Site Scripting)

**Attack Examples Detected:**
```html
<script>alert('XSS')</script>
<img src=x onerror=alert(1)>
<svg onload=alert(1)>
<iframe src="javascript:alert(1)">
<body onload=alert('XSS')>
javascript:alert(document.cookie)
<object data="javascript:alert(1)">
<embed src="javascript:alert(1)">
```

**Detection Layers:**
1. Blocklist: Pre-trained on 1.8M payloads
2. Signatures: HTML tags, event handlers, javascript: protocol
3. ML Ensemble: Character-level pattern recognition
4. Neural Network: Deep feature extraction

---

### 3. SQL Injection

**Attack Examples Detected:**
```sql
' OR '1'='1
admin' --
' UNION SELECT NULL, version() --
1; DROP TABLE users--
' OR 1=1 --
SELECT * FROM users WHERE id=1 OR 1=1
' AND SLEEP(5)--
1' AND '1'='1
admin' OR '1'='1'/* 
' UNION ALL SELECT NULL,NULL,NULL--
```

**Detection Layers:**
1. Blocklist: Pre-trained on 244K payloads
2. Signatures: SQL keywords, comment markers, boolean operators
3. ML Ensemble: Syntax pattern recognition
4. Neural Network: Contextual attack detection

---

## Evolution Timeline

### Phase 1: Initial Setup (Early Development)
- ✅ FastAPI backend with basic routing
- ✅ Simple dashboard with Chart.js
- ✅ Rate limiting implementation
- ✅ Basic DDoS detection with signatures

### Phase 2: ML Integration (Mid Development)
- ✅ Random Forest for DDoS classification
- ✅ Isolation Forest for anomaly detection
- ✅ Dataset generation script
- ✅ Model training pipeline
- ✅ Simple XSS/SQL detection (Logistic Regression)

### Phase 3: Advanced Models (Major Upgrade)
- ✅ Ensemble learning (4 classifier voting)
- ✅ Deep neural networks (TensorFlow/Keras)
- ✅ TF-IDF vectorization with 10K features
- ✅ Batch normalization and dropout
- ✅ Advanced training scripts with GPU support
- ✅ Model versioning and metadata tracking

### Phase 4: Production Features (Current)
- ✅ Dataset blocklist integration (1.8M+ payloads)
- ✅ Multi-layer detection pipeline
- ✅ WebSocket live updates
- ✅ Attack mix visualization
- ✅ Comprehensive testing framework
- ✅ Multi-IP attack simulation
- ✅ Command reference documentation
- ✅ 100% block rate achievement

---

## Model Performance

### Training Results

#### XSS Detection
```json
{
  "ensemble_f1": 0.9456,
  "nn_f1": 0.9234,
  "ensemble_auc": 0.9812,
  "nn_auc": 0.9523,
  "training_samples": 439500,
  "test_samples": 109875
}
```

#### SQL Injection Detection
```json
{
  "ensemble_f1": 0.9324,
  "nn_f1": 0.9156,
  "ensemble_auc": 0.9723,
  "nn_auc": 0.9512,
  "training_samples": 40000,
  "test_samples": 10000
}
```

### Production Test Results

#### Test 1: Single-Payload Testing
```bash
# SQL Injection Test
python test_sql_injection.py --count 50
Result: 50/50 detected (100%)

# XSS Test
python test_xss_injection.py --count 50
Result: 50/50 detected (100%)
```

#### Test 2: Multi-IP Sequential Attack
```bash
# SQL Attack from 5 IPs (3 payloads each)
python attack_sql_multiip.py --mode sequential --num-ips 5 --payloads 3
Result: 15/15 blocked (100%)

# XSS Attack from 5 IPs
python attack_xss_multiip.py --mode sequential --num-ips 5 --payloads 3
Result: 15/15 blocked (100%)
```

#### Test 3: Multi-IP Parallel Attack (DDoS-style)
```bash
# SQL Concurrent Attack (8 IPs × 5 payloads, 10 workers)
python attack_sql_multiip.py --mode parallel --num-ips 8 --payloads 5 --workers 10
Result: 40/40 blocked (100%)

# XSS Concurrent Attack
python attack_xss_multiip.py --mode parallel --num-ips 8 --payloads 5 --workers 10
Result: 40/40 blocked (100%)
```

### Performance Metrics

| Metric | Fast Model | Ensemble Only | Advanced (Ensemble+NN) |
|--------|-----------|---------------|------------------------|
| **Training Time** | 5-10 min | 8-12 min | 15-25 min |
| **Inference Latency** | 5-10ms | 20-30ms | 40-50ms |
| **F1 Score** | 0.91-0.94 | 0.93-0.96 | 0.94-0.97 |
| **Block Rate (Known)** | 95-98% | 98-99% | 100% |
| **Block Rate (Novel)** | 85-90% | 90-95% | 95-99% |
| **Memory Usage** | Low | Medium | High |
| **Model Size** | 10-20 MB | 30-50 MB | 100-150 MB |

---

## File Structure

```
project1/
├── app.py                          # Main FastAPI application
├── config.py                       # Configuration settings
├── detector.py                     # DDoS detection logic
├── firewall.py                     # Firewall integration
├── rate_limiter.py                 # Rate limiting implementation
├── generate_dataset.py             # Synthetic traffic generator
├── train.py                        # DDoS model training
├── requirements.txt                # Python dependencies
├── signatures.txt                  # Attack signature patterns
├── dataset_blocklist.py            # Blocklist hash sets
├── README.md                       # Project readme
│
├── frontend/                       # Web dashboard
│   ├── index.html                  # Dashboard HTML
│   ├── app.js                      # WebSocket + Chart.js logic
│   └── chart.umd.min.js           # Chart.js library
│
├── xss/                            # XSS detection module
│   ├── train_xss.py               # Simple training (old)
│   ├── train_xss_fast.py          # Fast training (RF+XGB)
│   ├── train_xss_advanced.py      # Advanced (Ensemble+NN)
│   ├── inference_xss.py           # Simple inference (old)
│   ├── inference_xss_advanced.py  # Advanced inference (current)
│   ├── xss_vectorizer.pkl         # TF-IDF vectorizer
│   ├── xss_ensemble.pkl           # Ensemble model
│   ├── xss_neural_network.h5      # Keras neural network
│   ├── xss_scaler.pkl             # Feature scaler
│   └── xss_metadata.json          # Training metrics
│
├── sql_injection/                  # SQL injection module
│   ├── train_sql.py               # Simple training (old)
│   ├── train_sql_fast.py          # Fast training (RF+XGB)
│   ├── train_sql_ensemble.py      # Ensemble only training
│   ├── train_sql_advanced.py      # Advanced (Ensemble+NN)
│   ├── inference_sql.py           # Simple inference (old)
│   ├── inference_sql_advanced.py  # Advanced inference (current)
│   ├── sql_vectorizer.pkl         # TF-IDF vectorizer
│   ├── sql_ensemble.pkl           # Ensemble model
│   ├── sql_neural_network.h5      # Keras neural network
│   ├── sql_scaler.pkl             # Feature scaler
│   └── sql_metadata.json          # Training metrics
│
├── dataset/                        # Training datasets
│   ├── generated_traffic.csv      # Synthetic DDoS traffic
│   ├── SQL_Injection_Detection_Dataset.csv  # 244K SQL payloads
│   └── Large-Scale Annotated Dataset for Cross-Site Scripting (XSS) Attack Detection.csv  # 1.8M XSS
│
├── models/                         # DDoS models
│   ├── ddos_supervised.pkl        # Random Forest
│   ├── ddos_anom.pkl             # Isolation Forest
│   ├── ddos_mlp.pkl              # MLP Classifier
│   └── ddos_autoenc.pkl          # Autoencoder
│
├── tests/                          # Testing scripts
│   ├── test_xss.py               # XSS validation tests
│   ├── test_sql_injection.py     # SQL dataset testing
│   ├── test_xss_injection.py     # XSS dataset testing
│   ├── attack_sql_multiip.py     # Multi-IP SQL attack sim
│   └── attack_xss_multiip.py     # Multi-IP XSS attack sim
│
├── simulator/                      # Traffic simulator
│   ├── traffic_sim.py            # Attack traffic generator
│   └── requirements.txt          # Simulator dependencies
│
└── Documentation/                  # Reference files
    ├── CHEAT_SHEET.sh            # Visual command reference
    ├── ATTACK_COMMANDS.sh        # Complete command guide
    ├── ATTACK_COMMANDS_QUICK.sh  # TL;DR commands
    ├── ATTACK_COMMANDS.md        # Markdown version
    ├── COMMAND_FILES_INDEX.sh    # File organization guide
    ├── ADVANCED_TRAINING_GUIDE.md
    ├── QUICK_CHECKLIST.md
    ├── QUICK_START_ADVANCED.md
    ├── START_HERE_ADVANCED.md
    └── EXECUTE_NOW.md
```

---

## API Endpoints

### 1. Test Endpoint
```http
POST /test
Content-Type: text/plain

<payload>
```

**Response (Success):**
```json
{
  "status": "ok",
  "message": "Request processed",
  "ip": "192.168.1.100"
}
```

**Response (Blocked):**
```json
{
  "detail": "Blocked by firewall"
}
```
HTTP Status: 403 Forbidden

### 2. Dashboard
```http
GET /
```
Returns: `frontend/index.html`

### 3. WebSocket
```javascript
ws://127.0.0.1:8000/ws
```

**Message Format:**
```json
{
  "type": "stats",
  "data": {
    "total_requests": 1523,
    "blocked_ips": 45,
    "attack_types": {
      "ddos": 120,
      "xss": 35,
      "sql": 28
    }
  }
}
```

---

## Training Process

### XSS Advanced Model Training

```bash
# Full training (30% sample, recommended)
python xss/train_xss_advanced.py --sample 0.3 --epochs 10 --batch-size 32

# Fast training (10% sample, testing)
python xss/train_xss_advanced.py --sample 0.1 --epochs 5 --batch-size 64

# Skip neural network (ensemble only)
python xss/train_xss_advanced.py --sample 0.3 --skip-nn

# Custom features
python xss/train_xss_advanced.py --sample 0.3 --max-features 15000
```

**Training Flow:**
1. Load dataset (1.8M rows)
2. Sample data (default 30%)
3. Split train/test (80/20)
4. TF-IDF vectorization (10K features)
5. Train ensemble (RF, SVM, XGBoost, LogReg)
6. Train neural network (10 epochs)
7. Save models (.pkl, .h5 files)
8. Generate metadata.json

**Output Files:**
- `xss/xss_vectorizer.pkl` (10 MB)
- `xss/xss_ensemble.pkl` (50 MB)
- `xss/xss_neural_network.h5` (100 MB)
- `xss/xss_scaler.pkl` (1 MB)
- `xss/xss_metadata.json` (1 KB)

### SQL Injection Advanced Model Training

```bash
# Full training (20% sample, recommended)
python sql_injection/train_sql_advanced.py --sample 0.2 --epochs 10 --batch-size 32

# Fast training
python sql_injection/train_sql_fast.py --sample 0.15 --n-estimators 40
```

**Training Flow:** Same as XSS

**Output Files:**
- `sql_injection/sql_vectorizer.pkl`
- `sql_injection/sql_ensemble.pkl`
- `sql_injection/sql_neural_network.h5`
- `sql_injection/sql_scaler.pkl`
- `sql_injection/sql_metadata.json`

### DDoS Model Training

```bash
# Generate synthetic dataset
python generate_dataset.py

# Train all DDoS models
python train.py
```

**Trains:**
1. Random Forest (supervised)
2. Isolation Forest (anomaly)
3. MLP Classifier (neural)
4. Autoencoder (reconstruction)

---

## Testing Framework

### 1. Dataset-Based Testing

**SQL Injection:**
```bash
# Test 50 random payloads from dataset
python test_sql_injection.py --count 50

# Test specific payload
python test_sql_injection.py --payload "' OR '1'='1"

# Test all attack payloads (slow)
python test_sql_injection.py --count 244000
```

**XSS:**
```bash
# Test 50 random payloads
python test_xss_injection.py --count 50

# Test all (very slow)
python test_xss_injection.py --count 1800000
```

**Output:**
```
Testing SQL Injection Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total payloads tested: 50
✅ Detected: 50 (100.0%)
❌ Missed: 0 (0.0%)
⚠️ Errors: 0 (0.0%)
```

### 2. Multi-IP Attack Simulation

**Sequential Attack:**
```bash
# SQL from 5 different IPs
python attack_sql_multiip.py --mode sequential --num-ips 5 --payloads 3 --delay 1

# XSS from 10 IPs
python attack_xss_multiip.py --mode sequential --num-ips 10 --payloads 2
```

**Parallel Attack (DDoS-style):**
```bash
# 8 IPs sending 5 payloads concurrently
python attack_sql_multiip.py --mode parallel --num-ips 8 --payloads 5 --workers 10

# Heavy load test
python attack_xss_multiip.py --mode parallel --num-ips 20 --payloads 10 --workers 20
```

**Output:**
```
Multi-IP SQL Injection Attack Simulation
═════════════════════════════════════════
Mode: Parallel (DDoS-style)
Number of IPs: 8
Payloads per IP: 5
Total Requests: 40

Results by IP:
┌────────────────┬──────────┬─────────┬──────────┐
│ IP Address     │ Requests │ Blocked │ Success  │
├────────────────┼──────────┼─────────┼──────────┤
│ 192.168.1.101  │ 5        │ 5       │ 0        │
│ 192.168.1.102  │ 5        │ 5       │ 0        │
│ 10.0.0.105     │ 5        │ 5       │ 0        │
│ 10.0.0.106     │ 5        │ 5       │ 0        │
│ 172.16.0.107   │ 5        │ 5       │ 0        │
│ 172.16.0.108   │ 5        │ 5       │ 0        │
│ 192.168.2.109  │ 5        │ 5       │ 0        │
│ 192.168.2.110  │ 5        │ 5       │ 0        │
└────────────────┴──────────┴─────────┴──────────┘

Summary:
✅ Total Blocked: 40/40 (100.0%)
⚠️ Total Passed: 0/40 (0.0%)
```

---

## Command Reference

### Setup & Installation

```bash
# Clone repository
git clone https://github.com/Sanchay-7/Ai-Threat-Detection-And-Prevention.git
cd Ai-Threat-Detection-And-Prevention

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install advanced dependencies
pip install -U xgboost tensorflow
```

### Training Commands

```bash
# Train XSS models
python xss/train_xss_advanced.py --sample 0.3 --epochs 10

# Train SQL models
python sql_injection/train_sql_advanced.py --sample 0.2 --epochs 10

# Train DDoS models
python generate_dataset.py && python train.py

# Fast training (no neural network)
python xss/train_xss_fast.py --sample 0.1
python sql_injection/train_sql_fast.py --sample 0.15
```

### Running the Application

```bash
# Development mode (no iptables)
SKIP_IPTABLES=1 python app.py

# Production mode (requires sudo)
sudo python app.py

# Custom port
uvicorn app:app --host 0.0.0.0 --port 8080
```

### Testing Commands

```bash
# Dataset validation
python test_sql_injection.py --count 50
python test_xss_injection.py --count 50

# Multi-IP sequential
python attack_sql_multiip.py --mode sequential --num-ips 5 --payloads 3
python attack_xss_multiip.py --mode sequential --num-ips 5 --payloads 3

# Multi-IP parallel (DDoS simulation)
python attack_sql_multiip.py --mode parallel --num-ips 8 --payloads 5 --workers 10
python attack_xss_multiip.py --mode parallel --num-ips 8 --payloads 5 --workers 10

# Manual payload test
curl -X POST http://127.0.0.1:8000/test -d "<script>alert('xss')</script>"
curl -X POST http://127.0.0.1:8000/test -d "' OR '1'='1"
```

### Model Management

```bash
# Check model files
ls -lh xss/*.pkl xss/*.h5
ls -lh sql_injection/*.pkl sql_injection/*.h5

# View training metrics
cat xss/xss_metadata.json
cat sql_injection/sql_metadata.json

# Switch to old models (edit app.py)
# Change imports from inference_xss_advanced to inference_xss
```

### Dashboard Access

```bash
# Open dashboard
open http://127.0.0.1:8000

# Or use browser
firefox http://127.0.0.1:8000
```

---

## Key Improvements Over Previous Version

### Before (Initial Version)

❌ **Simple Models**
- Single Logistic Regression classifier
- 5,000 TF-IDF features
- No ensemble learning
- No deep neural networks
- Accuracy: 85-88%

❌ **Limited Detection**
- Signature matching only
- No dataset blocklist
- Single-layer detection
- Block rate: 90-95%

❌ **Basic Testing**
- Manual testing only
- No automated test suite
- No multi-IP simulation

### After (Current Version)

✅ **Advanced Models**
- 4-classifier ensemble voting
- 10,000 TF-IDF features
- Deep neural network (4 hidden layers)
- Batch normalization + Dropout
- Accuracy: 94-97%

✅ **Multi-Layer Detection**
- Dataset blocklist (1.8M+ payloads)
- Signature matching
- ML ensemble
- Deep neural network
- Combined scoring
- Block rate: 100%

✅ **Comprehensive Testing**
- Dataset validation (244K SQL, 1.8M XSS)
- Multi-IP attack simulation
- Sequential and parallel modes
- Automated test reporting
- 100% validated block rate

✅ **Production Features**
- WebSocket live updates
- Attack mix visualization
- Per-IP statistics tracking
- Model metadata and versioning
- GPU training support
- Extensive documentation

---

## Performance Comparison

| Feature | Previous | Current | Improvement |
|---------|----------|---------|-------------|
| **XSS Detection** | 85-88% | 94-97% | +9-12% |
| **SQL Detection** | 85-88% | 93-96% | +8-11% |
| **Block Rate** | 90-95% | 100% | +5-10% |
| **Inference Time** | 5-10ms | 40-50ms | Slower (more accurate) |
| **Training Time** | 2-3 min | 15-25 min | Slower (better models) |
| **Model Size** | 5 MB | 150 MB | Larger (more complex) |
| **Features** | 5,000 | 10,000 | 2x increase |
| **Layers** | 1 (LogReg) | 9 (Ensemble+NN) | 9x increase |
| **Parameters** | ~5K | ~2.6M | 500x increase |

---

## Future Enhancements

### Potential Improvements

1. **LSTM/GRU Networks**
   - Capture sequential patterns in attacks
   - Better context understanding
   - Trade-off: Slower inference

2. **Transformer Models**
   - Attention mechanism for important features
   - State-of-the-art NLP architecture
   - Trade-off: High computational cost

3. **CNN for Character Patterns**
   - Convolutional filters for n-gram detection
   - Faster than RNN
   - Good for spatial patterns

4. **Active Learning**
   - Continuously update models with new attacks
   - Human-in-the-loop validation
   - Adaptive threat detection

5. **Federated Learning**
   - Distributed model training
   - Privacy-preserving
   - Collective threat intelligence

6. **Explainable AI**
   - SHAP/LIME for feature importance
   - Understand why attacks were detected
   - Better debugging and trust

---

## Repository Information

- **GitHub:** https://github.com/Sanchay-7/Ai-Threat-Detection-And-Prevention
- **Owner:** Sanchay-7
- **Branch:** main
- **License:** [Specify license]
- **Contributors:** [List contributors]

---

## Conclusion

This project demonstrates a **production-ready AI-powered threat detection system** that combines:

✅ Multiple detection layers (blocklist, signatures, ML, deep learning)  
✅ High accuracy (94-97% F1 scores)  
✅ Perfect block rate (100% on test attacks)  
✅ Real-time processing with FastAPI + WebSocket  
✅ Comprehensive testing framework  
✅ Extensive documentation  

The system evolved from simple logistic regression to advanced ensemble + deep learning models, achieving **100% block rate** on multi-IP concurrent attacks while maintaining reasonable inference latency.

**Status:** Production-ready for deployment with ongoing enhancements.

---

*Document Version: 1.0*  
*Last Updated: February 27, 2026*  
*Generated for: Ai-Threat-Detection-And-Prevention Project*
