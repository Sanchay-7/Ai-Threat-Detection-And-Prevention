#!/bin/bash
################################################################################
#              COMMAND REFERENCE FILES INDEX & GUIDE
#         All files for easy access to attack and testing commands
################################################################################

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════════╗
║                    COMMAND REFERENCE FILES INDEX                          ║
║                   Easy Access Guide for All Commands                       ║
╚════════════════════════════════════════════════════════════════════════════╝

📂 COMMAND REFERENCE FILES CREATED:
═════════════════════════════════════════════════════════════════════════════

1. 📋 CHEAT_SHEET.sh (19 KB) - START HERE! ⭐
   ├─ Purpose: Quick visual reference with ASCII formatting
   ├─ Content: Organized command boxes by category
   ├─ Best for: Quick lookups, copy-paste ready
   ├─ Format: Visual boxes with hierarchy
   ├─ View with: bash CHEAT_SHEET.sh
   └─ Time to read: 2 minutes

2. 📖 ATTACK_COMMANDS_QUICK.sh (8.5 KB) - QUICK REFERENCE
   ├─ Purpose: TL;DR version with essentials only
   ├─ Content: Commands grouped by use case
   ├─ Best for: Users who just want the basics
   ├─ Format: Bullet points and sections
   ├─ View with: bash ATTACK_COMMANDS_QUICK.sh
   └─ Time to read: 1 minute

3. 📚 ATTACK_COMMANDS.sh (23 KB) - COMPLETE REFERENCE
   ├─ Purpose: Comprehensive guide with all details
   ├─ Content: 10 sections covering every scenario
   ├─ Sections:
   │  ├─ Section 1: Start Backend Server
   │  ├─ Section 2: Dataset-Based Injection Testing
   │  ├─ Section 3: Multi-IP Attack Simulation
   │  ├─ Section 4: Quick Test Suites
   │  ├─ Section 5: Custom Attack Scenarios
   │  ├─ Section 6: Reporting & Analysis
   │  ├─ Section 7: Performance Testing
   │  ├─ Section 8: Useful Utilities
   │  ├─ Section 9: One-Liner Commands
   │  └─ Section 10: Expected Results & Success Indicators
   ├─ Best for: Complete understanding of all commands
   ├─ View with: bash ATTACK_COMMANDS.sh
   └─ Time to read: 10 minutes

4. 📘 ATTACK_COMMANDS.md (11 KB) - MARKDOWN DOCUMENTATION
   ├─ Purpose: Markdown formatted for easy viewing
   ├─ Content: Same as ATTACK_COMMANDS.sh but in Markdown
   ├─ Best for: Opening in text editors or browsers
   ├─ Format: Markdown with code blocks
   ├─ View with: cat ATTACK_COMMANDS.md | less
   │           or open in any text editor
   └─ Time to read: 10 minutes

═════════════════════════════════════════════════════════════════════════════

🎯 QUICK START GUIDE:
═════════════════════════════════════════════════════════════════════════════

Step 1: Read one of these files
   ├─ In a hurry? → bash CHEAT_SHEET.sh (2 min)
   ├─ Need basics? → bash ATTACK_COMMANDS_QUICK.sh (1 min)
   ├─ Want details? → bash ATTACK_COMMANDS.sh (10 min)
   └─ Prefer markdown? → cat ATTACK_COMMANDS.md (10 min)

Step 2: Start the backend
   SKIP_IPTABLES=1 python3 app.py

Step 3: Copy a command and run it
   Example: python3 test_sql_injection.py --sample 50

Step 4: Check results
   Expected: 100% detection rate, 0% false positives

═════════════════════════════════════════════════════════════════════════════

📋 COMMAND CATEGORIES COVERED:
═════════════════════════════════════════════════════════════════════════════

✅ Backend Management
   • Start/stop backend server
   • Check backend status
   • Restart backend

✅ Dataset-Based Testing
   • SQL injection payload testing (244K payloads)
   • XSS injection payload testing (1.8M payloads)
   • Quick, medium, large, and full dataset tests
   • Custom delay and API endpoint options

✅ Multi-IP Attack Simulation
   • Sequential attacks from different IPs
   • Parallel/DDoS-style attacks with workers
   • Various IP counts and request frequencies
   • Per-IP statistics tracking

✅ Special Attack Scenarios
   • Slow stealth attacks
   • Distributed attacks
   • Burst attacks
   • Wave attacks with delays

✅ Complete Test Suites
   • Quick tests (5 minutes)
   • Medium tests (20 minutes)
   • Comprehensive tests (30+ minutes)

✅ Remote API Testing
   • Test against different hosts
   • Custom API endpoints
   • Multi-IP attacks on remote servers

✅ Utilities & Monitoring
   • Backend health checks
   • Process management
   • Logging and result saving
   • Performance measurement
   • Report generation

═════════════════════════════════════════════════════════════════════════════

📊 WHAT EACH FILE CONTAINS:
═════════════════════════════════════════════════════════════════════════════

CHEAT_SHEET.sh includes:
  ├─ Setup instructions
  ├─ Dataset test commands
  ├─ Multi-IP attack commands (sequential)
  ├─ Multi-IP attack commands (parallel)
  ├─ Complete test suite commands
  ├─ Special scenario commands
  ├─ Remote API testing commands
  ├─ Utility commands
  ├─ Useful combinations
  ├─ Quick reference table
  └─ Success indicators

ATTACK_COMMANDS.sh includes everything above PLUS:
  ├─ Detailed explanations for each section
  ├─ Performance testing commands
  ├─ Load testing scenarios
  ├─ Expected results documentation
  ├─ Troubleshooting guide
  └─ Tips & tricks

ATTACK_COMMANDS_QUICK.sh includes:
  ├─ Most essential commands only
  ├─ Organized by frequency of use
  ├─ Success indicators
  ├─ File references
  └─ Quick tips

ATTACK_COMMANDS.md is:
  ├─ Same content as ATTACK_COMMANDS.sh
  ├─ But in Markdown format
  ├─ With nice tables and formatting
  ├─ Perfect for text editors

═════════════════════════════════════════════════════════════════════════════

💡 SUGGESTED READING ORDER:
═════════════════════════════════════════════════════════════════════════════

First Time Users:
  1. Start backend: SKIP_IPTABLES=1 python3 app.py
  2. Read: CHEAT_SHEET.sh (visual reference)
  3. Copy-paste: A quick test command
  4. See results: Check detection rate output

Detailed Learning:
  1. Read: ATTACK_COMMANDS_QUICK.sh (1 min overview)
  2. Read: CHEAT_SHEET.sh (2 min visual reference)
  3. Read: ATTACK_COMMANDS.sh (10 min complete guide)
  4. Experiment: Try different commands

Quick Reference During Testing:
  1. Keep open: CHEAT_SHEET.sh (fastest lookup)
  2. For details: ATTACK_COMMANDS.md (in editor)
  3. Execute: Copy commands directly

═════════════════════════════════════════════════════════════════════════════

🚀 MOST USED COMMANDS (Copy-Paste Ready):
═════════════════════════════════════════════════════════════════════════════

Start Backend:
  SKIP_IPTABLES=1 python3 app.py

Quick SQL Test (50 payloads):
  python3 test_sql_injection.py --sample 50

Quick XSS Test (50 payloads):
  python3 test_xss_injection.py --sample 50

SQL Multi-IP Attack (5 IPs):
  python3 attack_sql_multiip.py --ips 5 --requests-per-ip 3 --mode sequential

XSS Multi-IP Attack (5 IPs):
  python3 attack_xss_multiip.py --ips 5 --requests-per-ip 3 --mode sequential

DDoS-Style Test (8 IPs, parallel):
  python3 attack_sql_multiip.py --ips 8 --requests-per-ip 5 --mode parallel --workers 10

Complete Quick Test Suite:
  python3 test_sql_injection.py --sample 50 && \
  python3 test_xss_injection.py --sample 50 && \
  python3 attack_sql_multiip.py --ips 5 --requests-per-ip 3 --mode sequential && \
  python3 attack_xss_multiip.py --ips 5 --requests-per-ip 3 --mode sequential

═════════════════════════════════════════════════════════════════════════════

📌 HOW TO USE THESE FILES:
═════════════════════════════════════════════════════════════════════════════

In Terminal:
  $ bash CHEAT_SHEET.sh          # View as formatted output
  $ bash ATTACK_COMMANDS.sh      # View complete guide
  $ bash ATTACK_COMMANDS_QUICK.sh # View quick reference

In Text Editor:
  $ cat ATTACK_COMMANDS.md       # View markdown version
  $ less ATTACK_COMMANDS.md      # Page through markdown
  # Open with your favorite editor (VS Code, nano, vim, etc.)

Copy Commands:
  1. Open CHEAT_SHEET.sh or ATTACK_COMMANDS_QUICK.sh
  2. Find the command you want
  3. Copy the command line
  4. Paste into your terminal
  5. Run it!

═════════════════════════════════════════════════════════════════════════════

✅ FILE LOCATIONS:
═════════════════════════════════════════════════════════════════════════════

All files are in: /home/babayaga/Desktop/project1/

List all command reference files:
  ls -lh ATTACK_COMMANDS* CHEAT_SHEET.sh

View file sizes:
  du -h ATTACK_COMMANDS* CHEAT_SHEET.sh

Search for a command in all files:
  grep "DDoS" ATTACK_COMMANDS* CHEAT_SHEET.sh

═════════════════════════════════════════════════════════════════════════════

🎓 LEARNING PATHS:
═════════════════════════════════════════════════════════════════════════════

Path 1: Visual Learner (5 minutes)
  → bash CHEAT_SHEET.sh
  → Copy a command from a box
  → Run it and see results

Path 2: Quick User (2 minutes)
  → bash ATTACK_COMMANDS_QUICK.sh
  → Find your scenario
  → Copy-paste and run

Path 3: Thorough User (15 minutes)
  → Read ATTACK_COMMANDS.sh section by section
  → Understand each command's purpose
  → Experiment with variations

Path 4: Documentation Reader (20 minutes)
  → Open ATTACK_COMMANDS.md in editor
  → Read markdown formatting
  → Follow examples and explanations

═════════════════════════════════════════════════════════════════════════════

📞 NEED HELP?
═════════════════════════════════════════════════════════════════════════════

Check if backend is running:
  curl -s http://127.0.0.1:8000/test -X POST -d 'test' && echo 'OK' || echo 'DOWN'

View script help:
  python3 test_sql_injection.py --help
  python3 attack_sql_multiip.py --help

View backend logs:
  # In the terminal where you started backend, press Ctrl+C to stop
  # Or check any log files created

Check for errors:
  python3 test_sql_injection.py --sample 10 2>&1 | tail -20

═════════════════════════════════════════════════════════════════════════════

🎉 YOU'RE ALL SET!
═════════════════════════════════════════════════════════════════════════════

Now you have 4 comprehensive command reference files for:
  ✅ Starting the backend
  ✅ Running dataset-based tests
  ✅ Simulating multi-IP attacks
  ✅ Testing DDoS resistance
  ✅ Generating reports
  ✅ Monitoring and debugging

Each file has been carefully organized for easy access and quick reference.

Pick your preferred format and start testing!

═════════════════════════════════════════════════════════════════════════════

╔════════════════════════════════════════════════════════════════════════════╗
║                    FILES SUMMARY:                                         ║
║  CHEAT_SHEET.sh (19 KB)                - Visual quick reference            ║
║  ATTACK_COMMANDS_QUICK.sh (8.5 KB)     - TL;DR version                    ║
║  ATTACK_COMMANDS.sh (23 KB)            - Complete reference               ║
║  ATTACK_COMMANDS.md (11 KB)            - Markdown documentation           ║
╚════════════════════════════════════════════════════════════════════════════╝

EOF
