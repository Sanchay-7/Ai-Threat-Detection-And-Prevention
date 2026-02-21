# AI Threat Detection - Command Reference

## Quick Start

### 1. Start Backend Server
```bash
SKIP_IPTABLES=1 python3 app.py
```

### 2. Verify Backend is Running
```bash
curl -s http://127.0.0.1:8000/test -X POST -d 'test' && echo 'OK' || echo 'DOWN'
```

---

## Dataset-Based Tests

### SQL Injection Tests
```bash
# Quick test (50 payloads)
python3 test_sql_injection.py --sample 50

# Medium test (100 payloads)
python3 test_sql_injection.py --sample 100

# Large test (500 payloads)
python3 test_sql_injection.py --sample 500

# Full dataset (all 244K payloads - ~30 mins)
python3 test_sql_injection.py
```

### XSS Injection Tests
```bash
# Quick test (50 payloads)
python3 test_xss_injection.py --sample 50

# Medium test (100 payloads)
python3 test_xss_injection.py --sample 100

# Large test (1000 payloads)
python3 test_xss_injection.py --sample 1000

# Full dataset (all 1.8M payloads - hours)
python3 test_xss_injection.py
```

---

## Multi-IP Attack Simulation

### Sequential Attacks (One at a time)

#### SQL Injection
```bash
# 5 different IPs, 3 requests each
python3 attack_sql_multiip.py --ips 5 --requests-per-ip 3 --mode sequential

# 10 different IPs, 5 requests each
python3 attack_sql_multiip.py --ips 10 --requests-per-ip 5 --mode sequential

# 20 different IPs
python3 attack_sql_multiip.py --ips 20 --requests-per-ip 3 --mode sequential

# With slower delay (0.5s between requests)
python3 attack_sql_multiip.py --ips 10 --requests-per-ip 5 --mode sequential --delay 0.5
```

#### XSS Injection
```bash
# 5 different IPs, 3 requests each
python3 attack_xss_multiip.py --ips 5 --requests-per-ip 3 --mode sequential

# 10 different IPs, 5 requests each
python3 attack_xss_multiip.py --ips 10 --requests-per-ip 5 --mode sequential

# 20 different IPs
python3 attack_xss_multiip.py --ips 20 --requests-per-ip 3 --mode sequential
```

### Parallel/DDoS-Style Attacks

#### SQL Injection
```bash
# 8 IPs, 5 requests each, 10 concurrent threads
python3 attack_sql_multiip.py --ips 8 --requests-per-ip 5 --mode parallel --workers 10

# 15 IPs, 4 requests each, 15 workers
python3 attack_sql_multiip.py --ips 15 --requests-per-ip 4 --mode parallel --workers 15

# Heavy DDoS: 20 IPs × 10 requests × 20 workers
python3 attack_sql_multiip.py --ips 20 --requests-per-ip 10 --mode parallel --workers 20

# Extreme stress test: 50 IPs × 20 requests × 50 workers
python3 attack_sql_multiip.py --ips 50 --requests-per-ip 20 --mode parallel --workers 50
```

#### XSS Injection
```bash
# 8 IPs, 5 requests each, 10 concurrent threads
python3 attack_xss_multiip.py --ips 8 --requests-per-ip 5 --mode parallel --workers 10

# 15 IPs, 4 requests each, 15 workers
python3 attack_xss_multiip.py --ips 15 --requests-per-ip 4 --mode parallel --workers 15

# Heavy DDoS: 20 IPs × 10 requests × 20 workers
python3 attack_xss_multiip.py --ips 20 --requests-per-ip 10 --mode parallel --workers 20
```

---

## Complete Test Suites

### Quick Validation (5 minutes)
```bash
python3 test_sql_injection.py --sample 50 && \
python3 test_xss_injection.py --sample 50 && \
python3 attack_sql_multiip.py --ips 5 --requests-per-ip 3 --mode sequential && \
python3 attack_xss_multiip.py --ips 5 --requests-per-ip 3 --mode sequential && \
echo '=== ALL TESTS PASSED ==='
```

### Medium Suite (20 minutes)
```bash
python3 test_sql_injection.py --sample 100 && \
python3 test_xss_injection.py --sample 100 && \
python3 attack_sql_multiip.py --ips 10 --requests-per-ip 5 --mode parallel --workers 10 && \
python3 attack_xss_multiip.py --ips 10 --requests-per-ip 5 --mode parallel --workers 10 && \
echo '=== COMPREHENSIVE TEST COMPLETE ==='
```

### Full Validation (30+ minutes)
```bash
python3 test_sql_injection.py --sample 500 && \
python3 test_xss_injection.py --sample 500 && \
python3 attack_sql_multiip.py --ips 15 --requests-per-ip 4 --mode parallel --workers 15 && \
python3 attack_xss_multiip.py --ips 15 --requests-per-ip 4 --mode parallel --workers 15 && \
echo '=== FULL VALIDATION COMPLETE ==='
```

---

## Advanced Attack Scenarios

### Slow & Stealth Attack
```bash
# Single IP, many requests, slow delay (1 req/sec)
python3 attack_sql_multiip.py --ips 1 --requests-per-ip 30 --mode sequential --delay 1.0
```

### Distributed Attack
```bash
# Many IPs, few requests each
python3 attack_sql_multiip.py --ips 50 --requests-per-ip 1 --mode parallel --workers 50
```

### Burst Attack
```bash
# Few IPs, many concurrent requests
python3 attack_sql_multiip.py --ips 3 --requests-per-ip 20 --mode parallel --workers 20
```

### Wave Attack
```bash
# Simulate multiple attack waves with delay
python3 attack_sql_multiip.py --ips 10 --requests-per-ip 10 --mode parallel --workers 10 && \
sleep 10 && \
python3 attack_sql_multiip.py --ips 10 --requests-per-ip 10 --mode parallel --workers 10
```

---

## Custom API Targets

### Test Against Remote Server
```bash
# SQL test against custom API
python3 test_sql_injection.py --api-url http://192.168.1.100:8000/test --sample 50

# XSS test against custom API
python3 test_xss_injection.py --api-url http://remote-server:8000/test --sample 50

# Multi-IP attack against custom API
python3 attack_sql_multiip.py --api-url http://192.168.1.100:8000/test --ips 10 --requests-per-ip 5 --mode parallel
```

---

## Utility Commands

### Backend Management
```bash
# Check backend status
curl -s http://127.0.0.1:8000/test -X POST -d 'test' && echo 'Backend OK' || echo 'Backend DOWN'

# Check listening ports
lsof -i :8000

# Kill backend
pkill -f 'python app.py'

# Restart backend
pkill -f 'python app.py' && sleep 2 && SKIP_IPTABLES=1 python3 app.py &
```

### Logging & Results
```bash
# Save SQL test results
python3 test_sql_injection.py --sample 100 > sql_test_results.txt 2>&1

# Save XSS test results
python3 test_xss_injection.py --sample 100 > xss_test_results.txt 2>&1

# Save multi-IP results with timestamp
python3 attack_sql_multiip.py --ips 10 --requests-per-ip 5 --mode parallel > results_$(date +%s).txt 2>&1

# View results
cat sql_test_results.txt | less
```

### Performance Testing
```bash
# Measure SQL test execution time
time python3 test_sql_injection.py --sample 100

# Measure attack processing speed
time python3 attack_sql_multiip.py --ips 10 --requests-per-ip 5 --mode sequential

# Measure parallel attack speed
time python3 attack_sql_multiip.py --ips 20 --requests-per-ip 5 --mode parallel --workers 20
```

---

## Reporting

### Generate Reports
```bash
# Generate multi-IP attack report
python3 generate_multiip_report.py

# View the summary
python3 MULTIIP_TEST_SUMMARY.py

# View saved JSON report
cat multi_ip_attack_report.json | jq

# Pretty print with pagination
cat multi_ip_attack_report.json | jq | less
```

---

## Command-Line Options Reference

### All Test Scripts Support
| Option | Default | Description |
|--------|---------|-------------|
| `--api-url` | http://127.0.0.1:8000/test | API endpoint URL |
| `--sample` | (none) | Number of payloads to sample (omit for full dataset) |
| `--delay` | 0.05 | Delay between sequential requests (seconds) |

### Multi-IP Attack Scripts Support
| Option | Default | Description |
|--------|---------|-------------|
| `--api-url` | http://127.0.0.1:8000/test | API endpoint URL |
| `--ips` | 10 | Number of different source IPs |
| `--requests-per-ip` | 5 | Requests per IP |
| `--mode` | sequential | Attack mode (sequential or parallel) |
| `--workers` | 5 | Concurrent threads for parallel mode |
| `--delay` | 0.1 | Delay between sequential requests (seconds) |

---

## Expected Results

### ✅ Successful Dataset Tests
- Detection Rate: 100%
- No false positives detected
- All payloads processed successfully
- Response time < 100ms per request

### ✅ Successful Multi-IP Attacks
- Block rate: 100%
- All requests from different IPs blocked
- Per-IP statistics confirmed
- HTTP 403 responses for malicious payloads

### ✅ System Health Indicators
- No crashes or exceptions
- Proper error handling
- Firewall blocking all detected attacks
- Rate limiting working effectively

---

## Files Created for Testing

| File | Purpose |
|------|---------|
| `test_sql_injection.py` | SQL injection testing using dataset payloads |
| `test_xss_injection.py` | XSS injection testing using dataset payloads |
| `attack_sql_multiip.py` | SQL injection from multiple IP sources |
| `attack_xss_multiip.py` | XSS injection from multiple IP sources |
| `generate_multiip_report.py` | Generate comprehensive test report |
| `MULTIIP_TEST_SUMMARY.py` | Display test summary |
| `ATTACK_COMMANDS.sh` | Complete command reference |
| `ATTACK_COMMANDS_QUICK.sh` | Quick reference (TL;DR) |
| `ATTACK_COMMANDS.md` | Markdown command reference (this file) |

---

## Troubleshooting

### Backend Won't Start
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill any existing process
pkill -f 'python app.py'

# Try starting again
SKIP_IPTABLES=1 python3 app.py
```

### Connection Refused Error
```bash
# Make sure backend is running
curl -s http://127.0.0.1:8000/test -X POST -d 'test'

# If not running, start it:
SKIP_IPTABLES=1 python3 app.py &
```

### Tests Not Detecting Payloads
```bash
# Check backend logs for errors
# Verify backend is responding correctly
curl -s http://127.0.0.1:8000/test -X POST -d "admin' OR '1'='1" | jq

# Verify dataset files exist
ls -lh dataset/
```

---

## Tips & Tricks

1. **Run Multiple Tests in Parallel**: Use `&` to run commands in background
   ```bash
   python3 test_sql_injection.py --sample 100 & python3 test_xss_injection.py --sample 100 & wait
   ```

2. **Save Output with Timestamp**: 
   ```bash
   python3 test_sql_injection.py --sample 100 | tee log_$(date +%Y%m%d_%H%M%S).txt
   ```

3. **Monitor System Resources**:
   ```bash
   watch -n 1 'ps aux | grep python'
   ```

4. **Count Attack Requests**:
   ```bash
   # Quick calculation: IPs × Requests per IP
   # Example: 8 IPs × 5 requests = 40 total requests
   python3 attack_sql_multiip.py --ips 8 --requests-per-ip 5 --mode parallel --workers 8
   ```

5. **Test Different API Endpoints**:
   ```bash
   python3 test_sql_injection.py --api-url http://localhost:8000/test --sample 50
   python3 test_sql_injection.py --api-url http://192.168.1.1:8000/test --sample 50
   ```

---

## Support

For issues or questions:
- Check backend logs: `tail -f backend.log`
- Verify API is responding: `curl http://127.0.0.1:8000/test`
- Review individual script help: `python3 <script> --help`
- Repository: https://github.com/Sanchay-7/Ai-Threat-Detection-And-Prevention
