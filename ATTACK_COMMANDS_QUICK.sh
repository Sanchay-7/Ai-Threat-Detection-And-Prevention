#!/bin/bash
################################################################################
#                         QUICK COMMAND REFERENCE
#                           (TL;DR VERSION)
################################################################################

echo "
╔════════════════════════════════════════════════════════════════════════════╗
║                    AI THREAT DETECTION - QUICK COMMANDS                    ║
╚════════════════════════════════════════════════════════════════════════════╝

📌 QUICK COPY-PASTE COMMANDS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1️⃣  START BACKEND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SKIP_IPTABLES=1 python3 app.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2️⃣  DATASET-BASED TESTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SQL Injection Test (50 payloads):
  python3 test_sql_injection.py --sample 50

XSS Injection Test (50 payloads):
  python3 test_xss_injection.py --sample 50

SQL Full Dataset (all 244K payloads):
  python3 test_sql_injection.py

XSS Full Dataset (all 1.8M payloads):
  python3 test_xss_injection.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3️⃣  MULTI-IP ATTACKS (Sequential)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SQL Attack from 5 IPs:
  python3 attack_sql_multiip.py --ips 5 --requests-per-ip 3 --mode sequential

XSS Attack from 5 IPs:
  python3 attack_xss_multiip.py --ips 5 --requests-per-ip 3 --mode sequential

SQL Attack from 20 IPs:
  python3 attack_sql_multiip.py --ips 20 --requests-per-ip 3 --mode sequential

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4️⃣  MULTI-IP ATTACKS (Parallel/DDoS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SQL DDoS Attack (8 IPs, 5 requests each, 10 workers):
  python3 attack_sql_multiip.py --ips 8 --requests-per-ip 5 --mode parallel --workers 10

XSS DDoS Attack (8 IPs, 5 requests each, 10 workers):
  python3 attack_xss_multiip.py --ips 8 --requests-per-ip 5 --mode parallel --workers 10

Heavy DDoS (20 IPs, 10 requests, 20 workers):
  python3 attack_sql_multiip.py --ips 20 --requests-per-ip 10 --mode parallel --workers 20

Extreme Load (50 IPs, 20 requests, 50 workers):
  python3 attack_sql_multiip.py --ips 50 --requests-per-ip 20 --mode parallel --workers 50

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5️⃣  QUICK TEST SUITES (Run multiple tests)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Quick Test (5 mins):
  python3 test_sql_injection.py --sample 50 && python3 test_xss_injection.py --sample 50 && python3 attack_sql_multiip.py --ips 5 --requests-per-ip 3 --mode sequential && python3 attack_xss_multiip.py --ips 5 --requests-per-ip 3 --mode sequential

Complete Test (20 mins):
  python3 test_sql_injection.py --sample 100 && python3 test_xss_injection.py --sample 100 && python3 attack_sql_multiip.py --ips 10 --requests-per-ip 5 --mode parallel --workers 10 && python3 attack_xss_multiip.py --ips 10 --requests-per-ip 5 --mode parallel --workers 10

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6️⃣  UTILITY COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Check if backend is running:
  curl -s http://127.0.0.1:8000/test -X POST -d 'test' && echo 'Backend OK' || echo 'Backend DOWN'

Kill backend:
  pkill -f 'python app.py'

Restart backend:
  pkill -f 'python app.py' && sleep 2 && SKIP_IPTABLES=1 python3 app.py &

Check listening ports:
  lsof -i :8000

View report:
  python3 generate_multiip_report.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7️⃣  WITH CUSTOM OPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test with custom API (remote host):
  python3 test_sql_injection.py --api-url http://192.168.1.100:8000/test --sample 50

Attack custom API from multiple IPs:
  python3 attack_sql_multiip.py --api-url http://192.168.1.100:8000/test --ips 10 --requests-per-ip 5 --mode parallel

Slow stealth attack (1 request per second):
  python3 attack_sql_multiip.py --ips 1 --requests-per-ip 30 --mode sequential --delay 1.0

Burst attack with slow responses:
  python3 attack_sql_multiip.py --ips 5 --requests-per-ip 10 --mode parallel --workers 5 --delay 0.5

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ EXPECTED SUCCESS INDICATORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Dataset Tests:
   ✓ Detection Rate: 100%
   ✓ No false positives
   ✓ All payloads processed

✅ Multi-IP Attacks:
   ✓ Block rate: 100%
   ✓ Per-IP blocking confirmed
   ✓ DDoS resistance validated

✅ System Health:
   ✓ No crashes
   ✓ Response time < 100ms
   ✓ Proper error handling

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 FILES REFERENCE:
   test_sql_injection.py        - SQL injection dataset tests
   test_xss_injection.py        - XSS injection dataset tests
   attack_sql_multiip.py        - SQL multi-IP attack simulator
   attack_xss_multiip.py        - XSS multi-IP attack simulator
   ATTACK_COMMANDS.sh           - Complete command reference (this file)
   ATTACK_COMMANDS_QUICK.sh     - Quick reference (TL;DR)
   generate_multiip_report.py   - Report generator
   MULTIIP_TEST_SUMMARY.py      - Summary display

📚 For full details, see: ATTACK_COMMANDS.sh
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"
