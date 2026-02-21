#!/bin/bash
# COMMAND CHEAT SHEET - Copy & Paste Ready

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════════╗
║                   ATTACK COMMAND CHEAT SHEET                              ║
║                        Quick Reference Guide                              ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─ SETUP ────────────────────────────────────────────────────────────────────┐
│ # Start backend                                                            │
│ SKIP_IPTABLES=1 python3 app.py                                            │
│                                                                            │
│ # Test if running                                                          │
│ curl -s http://127.0.0.1:8000/test -X POST -d 'test' && echo 'OK'       │
└────────────────────────────────────────────────────────────────────────────┘

┌─ DATASET TESTS ────────────────────────────────────────────────────────────┐
│                                                                            │
│ SQL Injection:                                                             │
│ ├─ Quick:    python3 test_sql_injection.py --sample 50                   │
│ ├─ Medium:   python3 test_sql_injection.py --sample 100                  │
│ ├─ Large:    python3 test_sql_injection.py --sample 500                  │
│ └─ Full:     python3 test_sql_injection.py                               │
│                                                                            │
│ XSS Injection:                                                             │
│ ├─ Quick:    python3 test_xss_injection.py --sample 50                   │
│ ├─ Medium:   python3 test_xss_injection.py --sample 100                  │
│ ├─ Large:    python3 test_xss_injection.py --sample 1000                 │
│ └─ Full:     python3 test_xss_injection.py                               │
└────────────────────────────────────────────────────────────────────────────┘

┌─ MULTI-IP ATTACKS (Sequential) ────────────────────────────────────────────┐
│                                                                            │
│ SQL Injection:                                                             │
│ ├─ 5 IPs:    python3 attack_sql_multiip.py --ips 5 --requests-per-ip 3  │
│ ├─ 10 IPs:   python3 attack_sql_multiip.py --ips 10 --requests-per-ip 5 │
│ ├─ 20 IPs:   python3 attack_sql_multiip.py --ips 20 --requests-per-ip 3 │
│ └─ Custom:   python3 attack_sql_multiip.py --ips 50 --requests-per-ip 2 │
│                                                                            │
│ XSS Injection:                                                             │
│ ├─ 5 IPs:    python3 attack_xss_multiip.py --ips 5 --requests-per-ip 3  │
│ ├─ 10 IPs:   python3 attack_xss_multiip.py --ips 10 --requests-per-ip 5 │
│ └─ 20 IPs:   python3 attack_xss_multiip.py --ips 20 --requests-per-ip 3 │
└────────────────────────────────────────────────────────────────────────────┘

┌─ MULTI-IP ATTACKS (Parallel/DDoS) ─────────────────────────────────────────┐
│                                                                            │
│ SQL Injection (DDoS-style):                                                │
│ ├─ 8 IPs:    python3 attack_sql_multiip.py --ips 8 --requests-per-ip 5  │
│ │            --mode parallel --workers 10                                │
│ ├─ 15 IPs:   python3 attack_sql_multiip.py --ips 15 --requests-per-ip 4 │
│ │            --mode parallel --workers 15                                │
│ ├─ Heavy:    python3 attack_sql_multiip.py --ips 20 --requests-per-ip 10│
│ │            --mode parallel --workers 20                                │
│ └─ Extreme:  python3 attack_sql_multiip.py --ips 50 --requests-per-ip 20│
│              --mode parallel --workers 50                                 │
│                                                                            │
│ XSS Injection (DDoS-style):                                                │
│ ├─ 8 IPs:    python3 attack_xss_multiip.py --ips 8 --requests-per-ip 5  │
│ │            --mode parallel --workers 10                                │
│ └─ 15 IPs:   python3 attack_xss_multiip.py --ips 15 --requests-per-ip 4 │
│              --mode parallel --workers 15                                 │
└────────────────────────────────────────────────────────────────────────────┘

┌─ COMPLETE TEST SUITES ─────────────────────────────────────────────────────┐
│                                                                            │
│ Quick Test (5 min):                                                        │
│ python3 test_sql_injection.py --sample 50 && \                           │
│ python3 test_xss_injection.py --sample 50 && \                           │
│ python3 attack_sql_multiip.py --ips 5 --requests-per-ip 3 && \          │
│ python3 attack_xss_multiip.py --ips 5 --requests-per-ip 3               │
│                                                                            │
│ Complete Test (20 min):                                                    │
│ python3 test_sql_injection.py --sample 100 && \                          │
│ python3 test_xss_injection.py --sample 100 && \                          │
│ python3 attack_sql_multiip.py --ips 10 --requests-per-ip 5 --mode       │
│   parallel --workers 10 && \                                              │
│ python3 attack_xss_multiip.py --ips 10 --requests-per-ip 5 --mode       │
│   parallel --workers 10                                                    │
└────────────────────────────────────────────────────────────────────────────┘

┌─ SPECIAL SCENARIOS ────────────────────────────────────────────────────────┐
│                                                                            │
│ Slow Stealth Attack (1 request/sec):                                       │
│ python3 attack_sql_multiip.py --ips 1 --requests-per-ip 30              │
│         --mode sequential --delay 1.0                                     │
│                                                                            │
│ Distributed Attack (many IPs, few requests):                               │
│ python3 attack_sql_multiip.py --ips 50 --requests-per-ip 1              │
│         --mode parallel --workers 50                                      │
│                                                                            │
│ Burst Attack (few IPs, many concurrent requests):                          │
│ python3 attack_sql_multiip.py --ips 3 --requests-per-ip 20              │
│         --mode parallel --workers 20                                      │
│                                                                            │
│ Wave Attack (simulate attack waves with delay):                            │
│ python3 attack_sql_multiip.py --ips 10 --requests-per-ip 10 \           │
│         --mode parallel --workers 10 && sleep 10 && \                     │
│ python3 attack_sql_multiip.py --ips 10 --requests-per-ip 10 \           │
│         --mode parallel --workers 10                                      │
└────────────────────────────────────────────────────────────────────────────┘

┌─ REMOTE API TESTING ────────────────────────────────────────────────────────┐
│                                                                            │
│ Test against remote server:                                                │
│ python3 test_sql_injection.py                                             │
│   --api-url http://192.168.1.100:8000/test --sample 50                  │
│                                                                            │
│ Multi-IP attack on remote API:                                             │
│ python3 attack_sql_multiip.py                                             │
│   --api-url http://192.168.1.100:8000/test --ips 10 --requests-per-ip 5 │
│   --mode parallel --workers 10                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌─ UTILITIES ────────────────────────────────────────────────────────────────┐
│                                                                            │
│ Check backend status:                                                      │
│ curl -s http://127.0.0.1:8000/test -X POST -d 'test'                   │
│                                                                            │
│ Kill backend:                                                              │
│ pkill -f 'python app.py'                                                 │
│                                                                            │
│ Restart backend:                                                           │
│ pkill -f 'python app.py' && sleep 2 && SKIP_IPTABLES=1 python3 app.py & │
│                                                                            │
│ Generate report:                                                           │
│ python3 generate_multiip_report.py                                        │
│                                                                            │
│ View report:                                                               │
│ python3 MULTIIP_TEST_SUMMARY.py                                           │
│                                                                            │
│ Save test results:                                                         │
│ python3 test_sql_injection.py --sample 100 > results.txt 2>&1            │
│                                                                            │
│ Monitor test progress:                                                     │
│ watch -n 1 'ps aux | grep python'                                        │
└────────────────────────────────────────────────────────────────────────────┘

┌─ USEFUL COMBINATIONS ──────────────────────────────────────────────────────┐
│                                                                            │
│ Run SQL and XSS tests in parallel:                                         │
│ python3 test_sql_injection.py --sample 100 & \                           │
│ python3 test_xss_injection.py --sample 100 & wait                        │
│                                                                            │
│ Run all multi-IP attacks in parallel:                                      │
│ python3 attack_sql_multiip.py --ips 8 --requests-per-ip 5 \             │
│         --mode parallel --workers 8 & \                                    │
│ python3 attack_xss_multiip.py --ips 8 --requests-per-ip 5 \             │
│         --mode parallel --workers 8 & wait                                │
│                                                                            │
│ Measure execution time:                                                    │
│ time python3 test_sql_injection.py --sample 100                          │
│                                                                            │
│ Save with timestamp:                                                       │
│ python3 test_sql_injection.py --sample 100 > \                           │
│   results_$(date +%Y%m%d_%H%M%S).txt 2>&1                                │
└────────────────────────────────────────────────────────────────────────────┘

┌─ QUICK REFERENCE TABLE ────────────────────────────────────────────────────┐
│                                                                            │
│ Scenario                          Command                                 │
│ ─────────────────────────────────────────────────────────────────────     │
│ Quick sanity check (5 min)        Test 50 of each dataset                │
│ Validation test (20 min)          Test 100 of each + multi-IP attack     │
│ Comprehensive test (30+ min)      Test 500+ + stress test IPs            │
│ DDoS resistance test              Parallel attack with 20+ IPs           │
│ Rate limiting test                Multi-IP attack with different sources  │
│ API endpoint test                 Use --api-url with custom host         │
│ Performance benchmark             Use 'time' command with tests          │
│ Full dataset test                 Omit --sample flag (hours)             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌─ SUCCESS INDICATORS ────────────────────────────────────────────────────────┐
│                                                                            │
│ ✅ Dataset Tests:                                                          │
│    • Detection Rate: 100%                                                  │
│    • No false positives                                                    │
│    • All payloads processed                                                │
│                                                                            │
│ ✅ Multi-IP Attacks:                                                       │
│    • Block rate: 100%                                                      │
│    • HTTP 403 responses                                                    │
│    • Per-IP blocking confirmed                                             │
│                                                                            │
│ ✅ System Health:                                                          │
│    • No crashes                                                            │
│    • Response time < 100ms                                                 │
│    • Proper error handling                                                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════════════╗
║  For complete details, see: ATTACK_COMMANDS.sh or ATTACK_COMMANDS.md      ║
╚════════════════════════════════════════════════════════════════════════════╝

EOF
