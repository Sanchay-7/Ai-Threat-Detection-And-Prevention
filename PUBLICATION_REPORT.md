# Publication Report: AI-Powered Threat Detection and Prevention

**Project:** `Ai-Threat-Detection-And-Prevention`  
**Repository Owner:** `Sanchay-7`  
**Branch:** `main`  
**Date:** 4 March 2026

---

## 1) Abstract (Publication-Ready)

This project presents a real-time, hybrid web threat detection system that combines classical machine learning, deep learning, rule-based signatures, and dataset-derived blocklists to detect and block **DDoS**, **XSS**, and **SQL Injection** attacks. The system is implemented using **FastAPI** and supports live telemetry via **WebSocket** dashboards. A layered detection pipeline prioritizes deterministic detection of known malicious payloads, then applies statistical and neural scoring for generalized attack detection. Testing across sampled attack payloads and multi-IP concurrent simulations shows a high block rate (reported up to 100% in project test runs), with configurable thresholds and firewall integration for active mitigation.

---

## 2) Problem Statement and Motivation

Modern web systems face heterogeneous threats:
- volumetric/behavioral abuse (DDoS-like traffic patterns),
- payload-level application attacks (XSS, SQLi),
- and mixed evasive traffic.

Single-model defenses often fail under distribution shift or adversarial payload variation. This project addresses that by combining:
1. **hard-blocking** for known malicious strings,
2. **signature matching** for obvious exploits,
3. **ensemble ML** for robust probabilistic detection,
4. **deep neural scoring** for non-linear decision boundaries,
5. **rate limiting + firewalling** for mitigation.

---

## 3) System Architecture

### 3.1 High-Level Flow

1. Incoming request enters FastAPI middleware.
2. Rate limiter and existing firewall block state are checked.
3. Payload is evaluated by XSS and SQL advanced predictors.
4. Behavioral features are extracted for DDoS hybrid decisioning.
5. If malicious, IP is blocked (iptables if enabled) and events are logged.
6. Metrics/events are broadcast to dashboard clients via WebSockets.

### 3.2 Core Runtime Components
- `app.py`: middleware orchestration, feature extraction, endpoint handling, websocket broadcasting.
- `detector.py`: DDoS hybrid detector (RF + IF + MLP + autoencoder signals + signatures).
- `xss/inference_xss_advanced.py`: blocklist → regex signatures → ensemble+NN scoring.
- `sql_injection/inference_sql_advanced.py`: blocklist → regex signatures → ensemble+NN scoring.
- `rate_limiter.py`: per-IP token-bucket style controls.
- `firewall.py`: block/unblock integration (with optional skip mode for development).

---

## 4) Datasets and Feature Engineering

## 4.1 DDoS Dataset
- Source: synthetic generation from `generate_dataset.py`.
- Output: `dataset/generated_traffic.csv`.
- Features:
  - `req_rate`
  - `unique_paths_rate`
  - `ip_entropy`
  - `payload_size`
- Labels: `0` normal, `1` attack.

## 4.2 XSS Dataset
- File: `dataset/Large-Scale Annotated Dataset for Cross-Site Scripting (XSS) Attack Detection.csv`.
- Text columns normalized to `text`, labels to binary.
- Vectorization: **TF-IDF character n-grams (3,5)** up to 10,000 features.

## 4.3 SQL Injection Dataset
- File: `dataset/SQL_Injection_Detection_Dataset.csv`.
- Query column normalized to `text`, labels cleaned into binary.
- Vectorization: **TF-IDF character n-grams (3,5)** up to 10,000 features.

## 4.4 Blocklist Construction
- `dataset_blocklist.py` hashes known attack payloads from training datasets (MD5 hash lookup).
- Used as highest-priority deterministic detection stage.

---

## 5) Models Used (Previous vs Current)

## 5.1 Previous Baseline Models
- XSS/SQL simple pipelines (`train_xss.py`, `train_sql.py`):
  - TF-IDF + Logistic Regression.
- DDoS initial supervised/anomaly stack in `detector.py`.

## 5.2 Current Advanced Models (Active)

### XSS/SQL Ensemble
- `RandomForestClassifier`
- `LinearSVC` + `CalibratedClassifierCV` (probability calibration)
- `XGBoost` (`xgboost.XGBClassifier`)
- `LogisticRegression`
- Fusion: `VotingClassifier(voting='soft')`

### XSS/SQL Deep Neural Network
- Keras Sequential dense network:
  - Dense(256) + BatchNorm + Dropout(0.4)
  - Dense(128) + BatchNorm + Dropout(0.3)
  - Dense(64) + BatchNorm + Dropout(0.2)
  - Dense(32) + Dropout(0.1)
  - Dense(1, sigmoid)
- Optimizer: Adam (`lr=0.001`)
- Loss: binary crossentropy
- Metrics: accuracy + AUC

### DDoS Hybrid Models (`detector.py`)
- `RandomForestClassifier` (supervised)
- `IsolationForest` (anomaly)
- `MLPClassifier` pipeline with `StandardScaler`
- `MLPRegressor` autoencoder-like reconstruction pathway
- signature matching for known suspicious patterns

---

## 6) Important Code Snippets (for paper appendix)

> The snippets below are representative excerpts from project source files.

### 6.1 Advanced Inference Priority Logic (XSS/SQL)

```python
# Blocklist check (highest priority)
if check_xss_blocklist(payload):
    return {'decision': True, 'score': 1.0, 'reason': 'Blocked: Attack payload found in training dataset'}

# Signature check
for pattern in _COMPILED_PATTERNS:
    if pattern.search(payload):
        return {'decision': True, 'score': 0.99, 'reason': 'XSS signature match'}

# ML + NN scoring
ensemble_score = float(_ENSEMBLE_MODEL.predict_proba(X_tfidf)[0, 1])
neural_score = float(_NEURAL_MODEL.predict(X_scaled, verbose=0)[0, 0])
combined_score = (ensemble_score + neural_score) / 2
```

### 6.2 Ensemble Definition (Advanced Training)

```python
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)),
        ('svm', CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), method='sigmoid', cv=3)),
        ('xgb', xgb.XGBClassifier(n_estimators=150, max_depth=7, learning_rate=0.1, tree_method='hist')),
        ('lr', LogisticRegression(max_iter=500, n_jobs=-1, random_state=42))
    ],
    voting='soft'
)
```

### 6.3 Neural Network Architecture

```python
model = models.Sequential([
    layers.Dense(256, activation='relu', input_dim=input_dim),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(1, activation='sigmoid')
])
```

### 6.4 Real-time Middleware Detection (FastAPI)

```python
xss_result = predict_xss(payload_preview)
if xss_result["decision"]:
    block_ip_action(ip, xss_result["reason"], xss_result)
    return Response(content="XSS attack detected and IP blocked", status_code=403)

sql_result = predict_sql(payload_preview)
if sql_result["decision"]:
    block_ip_action(ip, sql_result["reason"], sql_result)
    return Response(content="SQL injection detected and IP blocked", status_code=403)
```

### 6.5 DDoS Feature Generation

```python
features = [
    req_rate,            # per-IP request frequency
    unique_paths_rate,   # path diversity
    ip_entropy,          # source distribution randomness
    payload_size         # body size heuristic
]
```

---

## 7) Requirements and Dependencies

## 7.1 Python Package Requirements (from `requirements.txt`)

```txt
fastapi>=0.95
uvicorn[standard]>=0.22
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
joblib>=1.3
python-multipart>=0.0.6
websockets>=11.0
aiofiles>=23.1
pytest>=7.4
```

## 7.2 Additional Advanced Model Dependencies
- `xgboost`
- `tensorflow`

> Note: advanced training/inference files import both libraries explicitly.

## 7.3 OS/Runtime Requirements
- Linux environment (for real `iptables` mode).
- Python 3.8+ (project has been used with newer Python as well).
- Sudo permissions if firewall mode is enabled.

---

## 8) Configuration and Operational Thresholds

From `config.py`:
- `WINDOW_SECONDS = 60`
- `ATTACK_BLOCK_SECONDS = 300`
- `DETECTION_THRESHOLD = 0.7`
- `MLP_THRESHOLD = 0.65`
- `AUTOENCODER_THRESHOLD = 0.7`
- `XSS_THRESHOLD = 0.55`
- `SQL_THRESHOLD = 0.55`

Model artifacts:
- DDoS models in `models/`
- XSS models in `xss/`
- SQL models in `sql_injection/`

---

## 9) Training and Reproducibility Protocol

## 9.1 DDoS Model Reproduction

```bash
python generate_dataset.py
python train.py
```

## 9.2 XSS Advanced Model Reproduction

```bash
python xss/train_xss_advanced.py --sample 0.3 --epochs 10 --batch-size 32
```

## 9.3 SQL Advanced Model Reproduction

```bash
python sql_injection/train_sql_advanced.py --sample 0.2 --epochs 10 --batch-size 32
```

## 9.4 Run API Service

```bash
SKIP_IPTABLES=1 python app.py
```

(or use firewall-enabled mode with proper sudo setup)

---

## 10) Evaluation and Validation Assets

Test/evaluation scripts present in repository:
- `test_sql_injection.py` (dataset-driven SQL detection checks)
- `test_xss_injection.py` (dataset-driven XSS detection checks)
- `attack_sql_multiip.py` (multi-IP sequential/parallel SQL simulation)
- `attack_xss_multiip.py` (multi-IP sequential/parallel XSS simulation)

Reported project testing outcomes (from project run history):
- SQL and XSS sampled detection runs showed very high detection.
- Multi-IP sequential and parallel attack simulations reported full blocking in tested scenarios.

> For publication, include your exact run logs/timestamps and hardware profile in supplementary material.

---

## 11) Publication Materials Checklist (What You Need)

Use this checklist before submission:

- [ ] **Paper draft** (IMRaD format: Introduction, Methods, Results, Discussion).
- [ ] **Problem formulation** and explicit threat model.
- [ ] **Dataset section** with source, preprocessing, class balance, split strategy.
- [ ] **Model section** (all architectures + hyperparameters).
- [ ] **Baselines** (e.g., LR-only, ensemble-only, NN-only).
- [ ] **Ablation study** (blocklist-only vs signature-only vs ML-only vs full stack).
- [ ] **Metrics**: Precision, Recall, F1, ROC-AUC, FPR/TPR, latency.
- [ ] **Operational metrics**: average inference time, memory use, throughput.
- [ ] **Reproducibility package**:
  - [ ] pinned dependencies,
  - [ ] training commands,
  - [ ] test scripts,
  - [ ] random seeds,
  - [ ] hardware/software versions.
- [ ] **Ethics and legal section** (authorized testing only, abuse prevention).
- [ ] **Limitations** (dataset shift, adversarial evasion, false-positive costs).
- [ ] **Future work** (continual learning, adversarial robustness, explainability).
- [ ] **Figures**:
  - [ ] system architecture diagram,
  - [ ] confusion matrices,
  - [ ] ROC curves,
  - [ ] detection latency chart.
- [ ] **Tables**:
  - [ ] model comparison table,
  - [ ] threshold sensitivity table,
  - [ ] ablation table.
- [ ] **Repository release tag** and artifact snapshot used for manuscript.

---

## 12) Suggested Paper Structure

1. **Title + Abstract**
2. **Introduction & Related Work**
3. **Threat Model and Scope**
4. **Methodology**
   - Data pipeline
   - Feature engineering
   - Hybrid detector design
5. **Implementation**
   - FastAPI runtime
   - middleware
   - firewall/rate limiting
6. **Experiments**
   - setup
   - baselines
   - metrics
7. **Results and Analysis**
8. **Limitations and Risks**
9. **Conclusion and Future Work**
10. **Reproducibility Appendix**

---

## 13) Risks, Limitations, and Responsible Use

- The system must be tested only on authorized infrastructure.
- Blocklist-centric gains can overestimate generalization; include out-of-distribution testing.
- Firewall actions in production should include rollback safeguards and monitoring.
- Threshold tuning should reflect cost asymmetry between false negatives and false positives.

---

## 14) Ready-to-Use Repro Commands (Quick Copy)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -U xgboost tensorflow
python generate_dataset.py
python train.py
python xss/train_xss_advanced.py --sample 0.3 --epochs 10 --batch-size 32
python sql_injection/train_sql_advanced.py --sample 0.2 --epochs 10 --batch-size 32
SKIP_IPTABLES=1 python app.py
```

---

## 15) What This Report Covers

This single file consolidates:
- project overview,
- previous and current model stacks,
- key implementation code excerpts,
- required dependencies and runtime constraints,
- reproducibility commands,
- publication checklist and manuscript structure.

It is intended to be directly used as the base document for preparing a conference/journal submission package.
