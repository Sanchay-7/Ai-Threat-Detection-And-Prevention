import time
import threading
import asyncio
import json
import logging
import math
from collections import defaultdict, deque
from typing import Dict, Deque, Any, List, Set, Tuple

from fastapi import FastAPI, Request, Response, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import config
from detector import HybridDetector
from rate_limiter import RateLimiterManager
from firewall import firewall
# ✅ ADVANCED MODELS - Uses Ensemble + Neural Networks
from xss.inference_xss_advanced import predict as predict_xss
from sql_injection.inference_sql_advanced import predict as predict_sql

# --- App Initialization ---
app = FastAPI(title="AI DDoS Hybrid Shield")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

# --- Global State & Models ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

lock = threading.RLock()
per_ip_requests: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=2000))
per_ip_paths: Dict[str, Deque[Tuple[float, str]]] = defaultdict(lambda: deque(maxlen=500))
global_requests: Deque[float] = deque(maxlen=20000)
global_ip_events: Deque[Tuple[float, str]] = deque(maxlen=20000)
global_ip_counts: Dict[str, int] = defaultdict(int)
event_log: Deque[Dict] = deque(maxlen=200)
attack_counts: Dict[str, int] = defaultdict(int)  # Track counts per attack type

detector = HybridDetector()
rate_mgr = RateLimiterManager()
_block_expirations: Dict[str, float] = {}

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
    while True:
        await asyncio.sleep(5)
        now = time.time()
        with lock:
            expired_ips = [ip for ip, expiry in _block_expirations.items() if now >= expiry]
        if expired_ips:
            for ip in expired_ips:
                if firewall.unblock_ip(ip):
                    with lock:
                        _block_expirations.pop(ip, None)
                    log_event("auto_unblock", ip, "Block duration expired")

async def broadcast_metrics():
    """Periodically computes and broadcasts metrics to all connected WebSocket clients."""
    while True:
        await asyncio.sleep(2) # Update interval
        if ws_manager.active_connections:
            metrics = get_metrics_data()
            await ws_manager.broadcast(json.dumps(metrics))

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up AI DDoS Shield...")
    detector.load_models()
    # Start all background tasks
    asyncio.create_task(unblock_expired_ips())
    asyncio.create_task(broadcast_metrics()) # Ensure live broadcast is started
    logger.info("Application startup complete.")

# --- Helper Functions ---
def get_client_ip(request: Request) -> str:
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for: return x_forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def prune_global_ip_state(now: float):
    while global_requests and global_requests[0] < now - config.WINDOW_SECONDS:
        global_requests.popleft()
    while global_ip_events and global_ip_events[0][0] < now - config.WINDOW_SECONDS:
        _, old_ip = global_ip_events.popleft()
        global_ip_counts[old_ip] -= 1
        if global_ip_counts[old_ip] <= 0:
            global_ip_counts.pop(old_ip, None)

def prune_ip_state(ip: str, now: float):
    while per_ip_requests[ip] and per_ip_requests[ip][0] < now - config.WINDOW_SECONDS:
        per_ip_requests[ip].popleft()
    while per_ip_paths[ip] and per_ip_paths[ip][0][0] < now - config.WINDOW_SECONDS:
        per_ip_paths[ip].popleft()

def compute_ip_entropy() -> float:
    total = sum(global_ip_counts.values())
    if total == 0:
        return 0.0
    entropy = -sum((count / total) * math.log2(count / total) for count in global_ip_counts.values())
    return float(entropy)

def log_event(kind: str, ip: str, reason: str, details: Any = None):
    details_text = details.get("details", "") if isinstance(details, dict) else ""
    event = {"ts": time.time(), "kind": kind, "ip": ip, "reason": reason, "details_text": details_text}
    event_log.appendleft(event)
    logger.info(f"EVENT: {kind} - IP: {ip} - Reason: {reason} - Details: {details_text}")
    # Track attack types
    if "xss" in reason.lower():
        attack_counts["xss"] += 1
    elif "sql" in reason.lower():
        attack_counts["sql"] += 1
    elif kind in {"auto_block", "manual_block"} and "xss" not in reason.lower() and "sql" not in reason.lower():
        attack_counts["ddos"] += 1


def block_ip_action(ip: str, reason: str, details: Dict):
    if ip in config.IP_WHITELIST:
        logger.warning(f"WHITELIST: Detected threat from {ip} but block was prevented.")
        return
    if not firewall.is_blocked(ip):
        if firewall.block_ip(ip):
            with lock:
                _block_expirations[ip] = time.time() + config.ATTACK_BLOCK_SECONDS
            log_event("auto_block", ip, reason, details)
        else:
            log_event("block_failed", ip, "iptables command failed", details)
    else:
        with lock:
            _block_expirations[ip] = time.time() + config.ATTACK_BLOCK_SECONDS
        logger.warning(f"IP {ip} already blocked. Extending block time.")

# --- Core Middleware ---
@app.middleware("http")
async def threat_detection_middleware(request: Request, call_next):
    # Allow static assets and health endpoints without inspection to avoid breaking the dashboard
    if request.url.path.startswith("/static") or request.url.path in {"/healthz", "/favicon.ico"}:
        return await call_next(request)

    ip = get_client_ip(request)
    if firewall.is_blocked(ip):
        return Response(content="Blocked by firewall", status_code=403)
    if not rate_mgr.allow(ip):
        log_event("rate_limit", ip, "Rate limit exceeded")
        return Response(content="Rate limit exceeded", status_code=429)

    now = time.monotonic()
    with lock:
        global_requests.append(now)
        global_ip_events.append((now, ip))
        global_ip_counts[ip] += 1
        per_ip_requests[ip].append(now)
        per_ip_paths[ip].append((now, request.url.path))
        prune_global_ip_state(now)
        prune_ip_state(ip, now)

        req_rate = len(per_ip_requests[ip]) / config.WINDOW_SECONDS
        unique_paths_rate = len({p for _, p in per_ip_paths[ip]}) / config.WINDOW_SECONDS
        ip_entropy = compute_ip_entropy()
        payload_size = int(request.headers.get('content-length', 0))
        features = [req_rate, unique_paths_rate, ip_entropy, payload_size]
    
    payload_preview = ""
    try: payload_preview = (await request.body())[:1024].decode("utf-8", errors="ignore")
    except Exception: pass
    
    # Check XSS
    xss_result = predict_xss(payload_preview)
    if xss_result["decision"]:
        block_ip_action(ip, xss_result["reason"], xss_result)
        if firewall.is_blocked(ip):
            return Response(content="XSS attack detected and IP blocked", status_code=403)
    
    # Check SQLi
    sql_result = predict_sql(payload_preview)
    if sql_result["decision"]:
        block_ip_action(ip, sql_result["reason"], sql_result)
        if firewall.is_blocked(ip):
            return Response(content="SQL injection detected and IP blocked", status_code=403)
        
    # Check DDoS
    result = detector.hybrid_decision(features, payload_preview)
    
    if result["decision"]:
        block_ip_action(ip, result["reason"], result)
        if firewall.is_blocked(ip):
            return Response(content="Threat detected and IP blocked", status_code=403)

    return await call_next(request)

# --- API and WebSocket Endpoints ---
@app.get("/", include_in_schema=False)
async def read_index():
    return FileResponse('frontend/index.html')

@app.get("/test-suite", include_in_schema=False)
async def read_test_suite():
    """Serve the security test suite with login form and attack testing"""
    return FileResponse('frontend/login.html')

def get_metrics_data() -> Dict:
    now_monotonic = time.monotonic()
    now_ts = time.time()
    active_ips_data = []
    with lock:
        # Cleanup old global requests
        prune_global_ip_state(now_monotonic)

        active_ips = list(per_ip_requests.keys())
        total_reqs_60s = len(global_requests)

        for ip in active_ips:
            ip_reqs = per_ip_requests.get(ip)
            if not ip_reqs or now_monotonic - ip_reqs[-1] > config.WINDOW_SECONDS * 2:
                if ip in per_ip_requests: del per_ip_requests[ip]
                if ip in per_ip_paths: del per_ip_paths[ip]
                continue

            prune_ip_state(ip, now_monotonic)
            req_rate_60s = len(ip_reqs) / config.WINDOW_SECONDS
            unique_paths_rate = len({p for _, p in per_ip_paths.get(ip, [])}) / config.WINDOW_SECONDS
            ip_entropy = compute_ip_entropy()
            features = [req_rate_60s, unique_paths_rate, ip_entropy, 0]
            decision = detector.hybrid_decision(features, "")
            
            reqs_10s = sum(1 for t in ip_reqs if t > now_monotonic - 10)
            share = (len(ip_reqs) / total_reqs_60s) * 100 if total_reqs_60s > 0 else 0

            active_ips_data.append({
                "ip": ip, "ip_rate_10s": round(reqs_10s / 10.0, 2),
                "proba": round(decision.get("score", 0.0), 2),
                "share": round(share, 1), "paths": len({p for _, p in per_ip_paths.get(ip, [])}),
                "is_blocked": firewall.is_blocked(ip)
            })
            
        blocked_list = [{"ip": k, "expires": int(v - now_ts)} for k, v in _block_expirations.items() if v > now_ts]
        event_list = list(event_log)
        ddos_count = attack_counts.get("ddos", 0)
        xss_count = attack_counts.get("xss", 0)
        sql_count = attack_counts.get("sql", 0)
    
    return {
        "total_req_rate_60s": round(len(global_requests) / config.WINDOW_SECONDS, 2),
        "total_req_rate_10s": round(sum(1 for t in global_requests if t > now_monotonic - 10) / 10.0, 2),
        "active_ips_count": len(active_ips_data),
        "top_talkers": sorted(active_ips_data, key=lambda x: x["ip_rate_10s"], reverse=True)[:50],
        "blocked_ips": sorted(blocked_list, key=lambda x: x["expires"]),
        "events": event_list,
        "attack_stats": {
            "ddos": ddos_count,
            "xss": xss_count,
            "sql": sql_count,
            "total": ddos_count + xss_count + sql_count
        }
    }

@app.get("/metrics")
async def metrics():
    return get_metrics_data() # For manual checking

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep connection alive
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

@app.post("/block")
async def manual_block(req: Dict[str, Any]):
    ip = req.get("ip")
    if not ip: raise HTTPException(status_code=400, detail="IP address required")
    block_ip_action(ip, "manual_block", {"source": "api"})
    return {"message": "Block request sent."}

@app.post("/unblock")
async def manual_unblock(req: Dict[str, Any]):
    ip = req.get("ip")
    if not ip: raise HTTPException(status_code=400, detail="IP address required")
    if firewall.unblock_ip(ip):
        with lock:
            _block_expirations.pop(ip, None)
        log_event("manual_unblock", ip, "Unblocked via API")
        return {"message": "Unblock request sent."}
    raise HTTPException(status_code=500, detail="Failed to unblock IP.")

@app.post("/test")
async def test_payload(request: Request):
    """Test endpoint for manual attack detection"""
    try:
        body = await request.body()
        payload = body.decode('utf-8')
        
        # Test XSS
        xss_result = predict_xss(payload)
        # Test SQL Injection
        sql_result = predict_sql(payload)
        
        client_ip = get_client_ip(request)
        
        # Log results
        if xss_result.get('decision'):
            log_event("auto_block", client_ip, f"XSS detected: {xss_result.get('reason', 'pattern match')}", {"score": xss_result.get('score')})
        if sql_result.get('decision'):
            log_event("auto_block", client_ip, f"SQL Injection detected: {sql_result.get('reason', 'pattern match')}", {"score": sql_result.get('score')})
        
        return {
            "payload": payload[:100],  # Return first 100 chars
            "xss": xss_result,
            "sql": sql_result,
            "tests_passed": (xss_result.get('decision'), sql_result.get('decision'))
        }
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
