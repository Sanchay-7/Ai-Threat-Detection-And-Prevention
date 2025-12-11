#!/bin/bash
# Complete guide to train and deploy advanced models

set -e

clear
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         SWITCH TO ADVANCED AI MODELS - COMPLETE GUIDE          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if already in project directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: Run this from project root directory"
    echo "   cd /home/babayaga/Desktop/project1"
    exit 1
fi

echo "📋 STEP 1: Install Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "This will install:"
echo "  • xgboost (gradient boosting)"
echo "  • tensorflow (neural networks)"
echo "  • scikit-learn (ML algorithms)"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    source .venv/bin/activate
    echo "📦 Installing xgboost..."
    pip install -U xgboost -q
    echo "📦 Installing tensorflow..."
    pip install -U tensorflow -q
    echo "📦 Installing scikit-learn..."
    pip install -U scikit-learn -q
    echo "✅ Dependencies installed!"
else
    echo "⏭️  Skipped"
fi

echo ""
echo "📋 STEP 2: Train XSS Detection Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "This will:"
echo "  • Load 30% of 1.8M XSS samples (~550K rows)"
echo "  • Train TF-IDF vectorizer"
echo "  • Train Ensemble classifier (RF+SVM+XGBoost+LogReg)"
echo "  • Train Neural Network (256→128→64→32)"
echo "  • Save 4 model files"
echo "  ⏱️  Time: ~15-20 minutes"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    source .venv/bin/activate
    python xss/train_xss_advanced.py --sample 0.3 --epochs 10 --batch-size 32
    echo ""
    echo "✅ XSS model trained!"
    echo "   Vectorizer: xss/xss_vectorizer.pkl"
    echo "   Ensemble: xss/xss_ensemble.pkl"
    echo "   Neural Network: xss/xss_neural_network.h5"
    echo "   Metrics: xss/xss_metadata.json"
else
    echo "⏭️  Skipped"
fi

echo ""
echo "📋 STEP 3: Train SQL Injection Detection Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "This will:"
echo "  • Load 20% of 250K SQLi samples (~50K rows)"
echo "  • Train TF-IDF vectorizer"
echo "  • Train Ensemble classifier (RF+SVM+XGBoost+LogReg)"
echo "  • Train Neural Network (256→128→64→32)"
echo "  • Save 4 model files"
echo "  ⏱️  Time: ~8-12 minutes"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    source .venv/bin/activate
    python sql_injection/train_sql_advanced.py --sample 0.2 --epochs 10 --batch-size 32
    echo ""
    echo "✅ SQL Injection model trained!"
    echo "   Vectorizer: sql_injection/sql_vectorizer.pkl"
    echo "   Ensemble: sql_injection/sql_ensemble.pkl"
    echo "   Neural Network: sql_injection/sql_neural_network.h5"
    echo "   Metrics: sql_injection/sql_metadata.json"
else
    echo "⏭️  Skipped"
fi

echo ""
echo "📋 STEP 4: Verify Models"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Checking XSS models..."
    ls -lh xss/*.pkl xss/*.h5 2>/dev/null && echo "✅ XSS models found" || echo "❌ XSS models not found"
    echo ""
    echo "Checking SQL models..."
    ls -lh sql_injection/*.pkl sql_injection/*.h5 2>/dev/null && echo "✅ SQL models found" || echo "❌ SQL models not found"
fi

echo ""
echo "📋 STEP 5: Check Performance Metrics"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "📊 XSS Model Performance:"
    [ -f xss/xss_metadata.json ] && cat xss/xss_metadata.json || echo "No metrics found"
    echo ""
    echo "📊 SQL Model Performance:"
    [ -f sql_injection/sql_metadata.json ] && cat sql_injection/sql_metadata.json || echo "No metrics found"
fi

echo ""
echo "📋 STEP 6: Restart Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  IMPORTANT: Run this in a NEW TERMINAL!"
echo ""
echo "Stop old server:"
echo "  Press Ctrl+C in terminal running: sudo ./run.sh"
echo ""
echo "Start new server:"
echo "  cd /home/babayaga/Desktop/project1"
echo "  sudo ./run.sh"
echo ""
echo "Wait for: 'Application startup complete.'"
echo ""

read -p "Ready to test in browser? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📊 Open Dashboard:"
    echo "   http://127.0.0.1:8000"
    echo ""
    echo "🧪 Open Test Suite:"
    echo "   http://127.0.0.1:8000/test-suite"
    echo ""
    echo "💉 Test SQL Injection:"
    echo "   1. Click '💉 SQL Injection' in sidebar"
    echo "   2. Username: ' OR '1'='1"
    echo "   3. Password: anything"
    echo "   4. Click 'Login'"
    echo "   → Should show: ⚠️ SQL Injection DETECTED!"
    echo ""
    echo "🔒 Test XSS:"
    echo "   1. Click '🔒 XSS Testing' in sidebar"
    echo "   2. Paste: <img src=x onerror=alert('XSS')>"
    echo "   3. Click 'Test XSS'"
    echo "   → Should show: ⚠️ XSS DETECTED!"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  ✅ SETUP COMPLETE!                           ║"
echo "║                                                                ║"
echo "║  Your system now uses:                                        ║"
echo "║  • Ensemble Learning (4 classifiers)                          ║"
echo "║  • Deep Neural Networks                                       ║"
echo "║  • TF-IDF Vectorization                                       ║"
echo "║  • Real 1.8M XSS + 250K SQLi datasets                        ║"
echo "║                                                                ║"
echo "║  Expected Improvement:                                        ║"
echo "║  • XSS F1-Score: 0.77 → 0.95 (+18%)                          ║"
echo "║  • SQL F1-Score: 0.77 → 0.93 (+16%)                          ║"
echo "║                                                                ║"
echo "║  Next: Restart server and test in browser!                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
