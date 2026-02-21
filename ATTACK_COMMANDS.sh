#!/bin/bash
################################################################################
#                    ATTACK & TESTING COMMAND REFERENCE
#                   All Commands for SQL/XSS Attack Testing
################################################################################
# This file contains all commands for running attack simulations and tests
# Usage: Copy and paste commands directly into your terminal
# Prerequisites: Backend running with SKIP_IPTABLES=1 python3 app.py
################################################################################

echo "
================================================================================
                    ATTACK & TESTING COMMAND REFERENCE
================================================================================
All commands are ready to copy and paste directly into your terminal.
Make sure backend is running first!
================================================================================
"

################################################################################
# SECTION 1: START BACKEND SERVER
################################################################################
echo "
================================================================================
SECTION 1: START BACKEND SERVER
================================================================================
"

echo "
1.1 Start Backend (FastAPI Server)
────────────────────────────────────────────────────────────────────────────"
echo "SKIP_IPTABLES=1 python3 app.py"

echo "
1.2 Start Backend with Python venv
────────────────────────────────────────────────────────────────────────────"
echo "cd /home/babayaga/Desktop/project1 && SKIP_IPTABLES=1 /home/babayaga/Desktop/project1/.venv/bin/python app.py"

echo "
1.3 Test if Backend is Running
────────────────────────────────────────────────────────────────────────────"
echo "curl -s http://127.0.0.1:8000/test -X POST -d 'test' | jq"

################################################################################
# SECTION 2: DATASET-BASED INJECTION TESTING
################################################################################
echo "
================================================================================
SECTION 2: DATASET-BASED INJECTION TESTING
================================================================================
Tests attacks using actual payloads from the training datasets
"

echo "
2.1 SQL Injection Testing (Dataset-based)
────────────────────────────────────────────────────────────────────────────"
echo "
Quick test (50 payloads):
  python3 test_sql_injection.py --sample 50

Medium test (100 payloads):
  python3 test_sql_injection.py --sample 100

Large test (500 payloads):
  python3 test_sql_injection.py --sample 500

Full dataset test (all 244K payloads - takes ~30 mins):
  python3 test_sql_injection.py

Custom delay between requests:
  python3 test_sql_injection.py --sample 100 --delay 0.2

Custom API endpoint:
  python3 test_sql_injection.py --api-url http://custom-host:8000/test --sample 50
"

echo "
2.2 XSS Injection Testing (Dataset-based)
────────────────────────────────────────────────────────────────────────────"
echo "
Quick test (50 payloads):
  python3 test_xss_injection.py --sample 50

Medium test (100 payloads):
  python3 test_xss_injection.py --sample 100

Large test (1000 payloads):
  python3 test_xss_injection.py --sample 1000

Full dataset test (all 1.8M payloads - takes hours):
  python3 test_xss_injection.py

Custom delay between requests:
  python3 test_xss_injection.py --sample 100 --delay 0.2

Custom API endpoint:
  python3 test_xss_injection.py --api-url http://custom-host:8000/test --sample 50
"

################################################################################
# SECTION 3: MULTI-IP ATTACK SIMULATION
################################################################################
echo "
================================================================================
SECTION 3: MULTI-IP ATTACK SIMULATION
================================================================================
Simulates attacks from multiple different source IP addresses
"

echo "
3.1 SQL Injection from Multiple IPs - Sequential Mode
────────────────────────────────────────────────────────────────────────────"
echo "
Attack from 5 different IPs (3 requests each):
  python3 attack_sql_multiip.py --ips 5 --requests-per-ip 3 --mode sequential

Attack from 10 different IPs (5 requests each):
  python3 attack_sql_multiip.py --ips 10 --requests-per-ip 5 --mode sequential

Attack from 10 IPs with slower delay (0.2s between requests):
  python3 attack_sql_multiip.py --ips 10 --requests-per-ip 5 --mode sequential --delay 0.2

Attack from 20 different IPs:
  python3 attack_sql_multiip.py --ips 20 --requests-per-ip 3 --mode sequential
"

echo "
3.2 SQL Injection from Multiple IPs - Parallel/DDoS Mode
────────────────────────────────────────────────────────────────────────────"
echo "
DDoS-style attack: 8 IPs × 5 requests with 10 concurrent threads:
  python3 attack_sql_multiip.py --ips 8 --requests-per-ip 5 --mode parallel --workers 10

DDoS-style attack: 15 IPs × 4 requests with 15 concurrent threads:
  python3 attack_sql_multiip.py --ips 15 --requests-per-ip 4 --mode parallel --workers 15

Heavy DDoS: 20 IPs × 10 requests with 20 concurrent threads:
  python3 attack_sql_multiip.py --ips 20 --requests-per-ip 10 --mode parallel --workers 20

Stress test: 30 IPs × 5 requests with 30 workers:
  python3 attack_sql_multiip.py --ips 30 --requests-per-ip 5 --mode parallel --workers 30
"

echo "
3.3 XSS Injection from Multiple IPs - Sequential Mode
────────────────────────────────────────────────────────────────────────────"
echo "
Attack from 5 different IPs (3 requests each):
  python3 attack_xss_multiip.py --ips 5 --requests-per-ip 3 --mode sequential

Attack from 10 different IPs (5 requests each):
  python3 attack_xss_multiip.py --ips 10 --requests-per-ip 5 --mode sequential

Attack from 10 IPs with slower delay (0.2s between requests):
  python3 attack_xss_multiip.py --ips 10 --requests-per-ip 5 --mode sequential --delay 0.2

Attack from 20 different IPs:
  python3 attack_xss_multiip.py --ips 20 --requests-per-ip 3 --mode sequential
"

echo "
3.4 XSS Injection from Multiple IPs - Parallel/DDoS Mode
────────────────────────────────────────────────────────────────────────────"
echo "
DDoS-style attack: 8 IPs × 5 requests with 10 concurrent threads:
  python3 attack_xss_multiip.py --ips 8 --requests-per-ip 5 --mode parallel --workers 10

DDoS-style attack: 15 IPs × 4 requests with 15 concurrent threads:
  python3 attack_xss_multiip.py --ips 15 --requests-per-ip 4 --mode parallel --workers 15

Heavy DDoS: 20 IPs × 10 requests with 20 concurrent threads:
  python3 attack_xss_multiip.py --ips 20 --requests-per-ip 10 --mode parallel --workers 20

Stress test: 30 IPs × 5 requests with 30 workers:
  python3 attack_xss_multiip.py --ips 30 --requests-per-ip 5 --mode parallel --workers 30
"

################################################################################
# SECTION 4: QUICK TEST SUITES
################################################################################
echo "
================================================================================
SECTION 4: QUICK TEST SUITES (Run Multiple Tests)
================================================================================
"

echo "
4.1 Quick Validation Suite (5 minutes)
────────────────────────────────────────────────────────────────────────────
Run this to quickly validate both detectors:
"
echo "
  echo '=== SQL Dataset Test ===' && python3 test_sql_injection.py --sample 50
  echo '=== XSS Dataset Test ===' && python3 test_xss_injection.py --sample 50
  echo '=== SQL Multi-IP Test ===' && python3 attack_sql_multiip.py --ips 5 --requests-per-ip 3 --mode sequential
  echo '=== XSS Multi-IP Test ===' && python3 attack_xss_multiip.py --ips 5 --requests-per-ip 3 --mode sequential
"

echo "
4.2 Medium Test Suite (20 minutes)
────────────────────────────────────────────────────────────────────────────
Run this for comprehensive testing:
"
echo "
  echo '=== SQL Dataset Test (100 payloads) ===' && python3 test_sql_injection.py --sample 100
  echo '=== XSS Dataset Test (100 payloads) ===' && python3 test_xss_injection.py --sample 100
  echo '=== SQL Parallel Attack ===' && python3 attack_sql_multiip.py --ips 8 --requests-per-ip 5 --mode parallel
  echo '=== XSS Parallel Attack ===' && python3 attack_xss_multiip.py --ips 8 --requests-per-ip 5 --mode parallel
"

echo "
4.3 Full Validation Suite (30+ minutes)
────────────────────────────────────────────────────────────────────────────
Run this for thorough system validation:
"
echo "
  echo '=== SQL Dataset Test (500 payloads) ===' && python3 test_sql_injection.py --sample 500
  echo '=== XSS Dataset Test (500 payloads) ===' && python3 test_xss_injection.py --sample 500
  echo '=== SQL Stress Test (15 IPs, 4 reqs each) ===' && python3 attack_sql_multiip.py --ips 15 --requests-per-ip 4 --mode parallel --workers 15
  echo '=== XSS Stress Test (15 IPs, 4 reqs each) ===' && python3 attack_xss_multiip.py --ips 15 --requests-per-ip 4 --mode parallel --workers 15
"

################################################################################
# SECTION 5: CUSTOM ATTACK SCENARIOS
################################################################################
echo "
================================================================================
SECTION 5: CUSTOM ATTACK SCENARIOS
================================================================================
"

echo "
5.1 Targeted Attack Scenarios
────────────────────────────────────────────────────────────────────────────"
echo "
Slow & Stealth Attack (Single IP, slow requests):
  python3 attack_sql_multiip.py --ips 1 --requests-per-ip 20 --mode sequential --delay 1.0

Distributed Attack (Many IPs, few requests each):
  python3 attack_sql_multiip.py --ips 50 --requests-per-ip 1 --mode parallel --workers 50

Burst Attack (Few IPs, many concurrent requests):
  python3 attack_sql_multiip.py --ips 3 --requests-per-ip 20 --mode parallel --workers 20

Wave Attack (Simulate multiple attack waves):
  python3 attack_sql_multiip.py --ips 10 --requests-per-ip 10 --mode parallel --workers 10 && sleep 10 && python3 attack_sql_multiip.py --ips 10 --requests-per-ip 10 --mode parallel --workers 10
"

echo "
5.2 API Endpoint Testing with Custom Host
────────────────────────────────────────────────────────────────────────────"
echo "
Test against remote API:
  python3 test_sql_injection.py --api-url http://192.168.1.100:8000/test --sample 50

Multi-IP attack against custom API:
  python3 attack_sql_multiip.py --api-url http://192.168.1.100:8000/test --ips 5 --requests-per-ip 3

XSS test against custom API:
  python3 test_xss_injection.py --api-url http://remote-server:8000/test --sample 50
"

################################################################################
# SECTION 6: REPORTING & ANALYSIS
################################################################################
echo "
================================================================================
SECTION 6: REPORTING & ANALYSIS
================================================================================
"

echo "
6.1 Generate Test Reports
────────────────────────────────────────────────────────────────────────────"
echo "
Generate Multi-IP Attack Report:
  python3 generate_multiip_report.py

View the Summary:
  python3 MULTIIP_TEST_SUMMARY.py

View saved JSON report:
  cat multi_ip_attack_report.json | jq

Pretty print with less:
  cat multi_ip_attack_report.json | jq | less
"

echo "
6.2 Save Test Results
────────────────────────────────────────────────────────────────────────────"
echo "
Save SQL test results to file:
  python3 test_sql_injection.py --sample 100 > sql_test_results.txt 2>&1

Save XSS test results to file:
  python3 test_xss_injection.py --sample 100 > xss_test_results.txt 2>&1

Save multi-IP test results:
  python3 attack_sql_multiip.py --ips 10 --requests-per-ip 5 --mode parallel > multiip_results.txt 2>&1

View results with timestamps:
  echo \"Test run at: \$(date)\" && python3 test_sql_injection.py --sample 50 | tee test_output_\$(date +%s).log
"

################################################################################
# SECTION 7: PERFORMANCE TESTING
################################################################################
echo "
================================================================================
SECTION 7: PERFORMANCE TESTING
================================================================================
"

echo "
7.1 Response Time Benchmarks
────────────────────────────────────────────────────────────────────────────"
echo "
Test with timing information:
  time python3 test_sql_injection.py --sample 100
  time python3 test_xss_injection.py --sample 100

Measure attack processing speed (SQL):
  time python3 attack_sql_multiip.py --ips 10 --requests-per-ip 5 --mode sequential

Measure attack processing speed (Parallel):
  time python3 attack_sql_multiip.py --ips 20 --requests-per-ip 5 --mode parallel --workers 20
"

echo "
7.2 Load Testing
────────────────────────────────────────────────────────────────────────────"
echo "
Light load (100 total requests):
  python3 attack_sql_multiip.py --ips 10 --requests-per-ip 10 --mode parallel

Medium load (300 total requests):
  python3 attack_sql_multiip.py --ips 15 --requests-per-ip 20 --mode parallel

Heavy load (500 total requests):
  python3 attack_sql_multiip.py --ips 25 --requests-per-ip 20 --mode parallel

Extreme load (1000 total requests - 50 workers):
  python3 attack_sql_multiip.py --ips 50 --requests-per-ip 20 --mode parallel --workers 50
"

################################################################################
# SECTION 8: USEFUL UTILITIES
################################################################################
echo "
================================================================================
SECTION 8: USEFUL UTILITIES & SHORTCUTS
================================================================================
"

echo "
8.1 Check Backend Status
────────────────────────────────────────────────────────────────────────────"
echo "
Check if backend is running:
  curl -s http://127.0.0.1:8000/test -X POST -d 'test' && echo 'Backend OK' || echo 'Backend DOWN'

Get backend response time:
  time curl -s http://127.0.0.1:8000/test -X POST -d 'test' > /dev/null

Check listening ports:
  netstat -tlnp | grep 8000
  lsof -i :8000
"

echo "
8.2 Kill & Restart Backend
────────────────────────────────────────────────────────────────────────────"
echo "
Kill all Python processes:
  pkill -f 'python app.py'

Kill specific FastAPI process:
  lsof -i :8000 | grep LISTEN | awk '{print \$2}' | xargs kill -9

Restart backend:
  pkill -f 'python app.py' && sleep 2 && SKIP_IPTABLES=1 python3 app.py &
"

echo "
8.3 Monitor Backend Logs
────────────────────────────────────────────────────────────────────────────"
echo "
Run backend with visible logs:
  SKIP_IPTABLES=1 python3 app.py

Run backend and save logs:
  SKIP_IPTABLES=1 python3 app.py > backend.log 2>&1 &

Tail backend logs in real-time:
  tail -f backend.log

Monitor system resources while testing:
  watch -n 1 'ps aux | grep python'
"

################################################################################
# SECTION 9: COMBINED COMMANDS (COPY & PASTE READY)
################################################################################
echo "
================================================================================
SECTION 9: ONE-LINER COMMANDS (Ready to Copy & Paste)
================================================================================
"

echo "
9.1 Complete Test Suite (Run all tests in sequence)
────────────────────────────────────────────────────────────────────────────"
echo "
# Quick 5-minute test
python3 test_sql_injection.py --sample 50 && python3 test_xss_injection.py --sample 50 && python3 attack_sql_multiip.py --ips 5 --requests-per-ip 3 --mode sequential && python3 attack_xss_multiip.py --ips 5 --requests-per-ip 3 --mode sequential && echo '=== ALL TESTS PASSED ==='

# Comprehensive 20-minute test
python3 test_sql_injection.py --sample 200 && python3 test_xss_injection.py --sample 200 && python3 attack_sql_multiip.py --ips 10 --requests-per-ip 5 --mode parallel --workers 10 && python3 attack_xss_multiip.py --ips 10 --requests-per-ip 5 --mode parallel --workers 10 && echo '=== COMPREHENSIVE TEST COMPLETE ==='
"

echo "
9.2 Parallel Test Execution
────────────────────────────────────────────────────────────────────────────"
echo "
# Run SQL and XSS tests in parallel
python3 test_sql_injection.py --sample 100 & python3 test_xss_injection.py --sample 100 & wait

# Run all multi-IP tests in parallel
python3 attack_sql_multiip.py --ips 8 --requests-per-ip 5 --mode parallel & python3 attack_xss_multiip.py --ips 8 --requests-per-ip 5 --mode parallel & wait
"

################################################################################
# SECTION 10: EXPECTED RESULTS & SUCCESS INDICATORS
################################################################################
echo "
================================================================================
SECTION 10: EXPECTED RESULTS & SUCCESS INDICATORS
================================================================================
"

echo "
✅ Dataset Tests Success Indicators:
   - 'Loaded X SQL injection payloads'
   - 'Detected: 50/50 (100.0%)'
   - 'Correct (not flagged): 50/50 (100.0%)'
   - 'Detection Rate: 100.0%'

✅ Multi-IP Attack Success Indicators:
   - 'IPs blocked: X' (should equal total requests)
   - 'Block rate: 100.0%'
   - 'Per-IP Statistics' table showing blocked count
   - HTTP 403 responses

⚠️  API Error Indicators:
   - 'Connection refused' = Backend not running
   - 'HTTP 403' = Attack blocked (expected)
   - 'JSON decode error' = Backend issue

✅ System is working correctly when:
   - All dataset tests show 100% detection
   - All multi-IP attacks show 100% block rate
   - No false positives on benign payloads
   - Response time < 100ms per request
"

################################################################################
# END OF COMMAND REFERENCE
################################################################################
echo "
================================================================================
                    END OF COMMAND REFERENCE
================================================================================
For more details, see individual script help:
  python3 test_sql_injection.py --help
  python3 test_xss_injection.py --help
  python3 attack_sql_multiip.py --help
  python3 attack_xss_multiip.py --help

Email: For issues or questions, contact the security team
Repository: https://github.com/Sanchay-7/Ai-Threat-Detection-And-Prevention
================================================================================
"
