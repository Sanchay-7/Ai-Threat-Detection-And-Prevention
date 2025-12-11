#!/usr/bin/env bash
# Verify advanced models are properly configured

echo "🔍 Verifying Advanced Model Setup"
echo "=================================="
echo ""

# Check 1: app.py imports
echo "✓ Check 1: app.py uses advanced models"
if grep -q "inference_xss_advanced" app.py && grep -q "inference_sql_advanced" app.py; then
    echo "  ✅ PASS: app.py imports advanced models"
else
    echo "  ❌ FAIL: app.py still uses old models"
    echo "  Fix: The app.py has been updated. Restart server."
fi

# Check 2: Advanced training scripts exist
echo ""
echo "✓ Check 2: Advanced training scripts exist"
if [ -f "xss/train_xss_advanced.py" ] && [ -f "sql_injection/train_sql_advanced.py" ]; then
    echo "  ✅ PASS: Training scripts found"
else
    echo "  ❌ FAIL: Training scripts not found"
fi

# Check 3: Advanced inference modules exist
echo ""
echo "✓ Check 3: Advanced inference modules exist"
if [ -f "xss/inference_xss_advanced.py" ] && [ -f "sql_injection/inference_sql_advanced.py" ]; then
    echo "  ✅ PASS: Inference modules found"
else
    echo "  ❌ FAIL: Inference modules not found"
fi

# Check 4: Dataset files exist
echo ""
echo "✓ Check 4: Dataset files exist"
xss_file="dataset/Large-Scale Annotated Dataset for Cross-Site Scripting (XSS) Attack Detection.csv"
sql_file="dataset/SQL_Injection_Detection_Dataset.csv"

if [ -f "$xss_file" ]; then
    xss_size=$(du -h "$xss_file" | cut -f1)
    echo "  ✅ PASS: XSS dataset found ($xss_size)"
else
    echo "  ❌ FAIL: XSS dataset not found"
fi

if [ -f "$sql_file" ]; then
    sql_size=$(du -h "$sql_file" | cut -f1)
    echo "  ✅ PASS: SQL dataset found ($sql_size)"
else
    echo "  ❌ FAIL: SQL dataset not found"
fi

# Check 5: Models directory
echo ""
echo "✓ Check 5: Model directories"
[ -d "xss" ] && echo "  ✅ xss/ directory exists" || echo "  ❌ xss/ not found"
[ -d "sql_injection" ] && echo "  ✅ sql_injection/ directory exists" || echo "  ❌ sql_injection/ not found"

# Check 6: Check for existing models
echo ""
echo "✓ Check 6: Existing trained models"
xss_models=$(find xss -name "*.pkl" -o -name "*.h5" 2>/dev/null | wc -l)
sql_models=$(find sql_injection -name "*.pkl" -o -name "*.h5" 2>/dev/null | wc -l)

if [ $xss_models -gt 0 ]; then
    echo "  ✅ XSS models found: $xss_models files"
else
    echo "  ⏳ XSS models not trained yet (will be created during training)"
fi

if [ $sql_models -gt 0 ]; then
    echo "  ✅ SQL models found: $sql_models files"
else
    echo "  ⏳ SQL models not trained yet (will be created during training)"
fi

# Summary
echo ""
echo "=================================="
echo "📋 NEXT STEPS:"
echo ""
echo "1. Install dependencies:"
echo "   source .venv/bin/activate"
echo "   pip install -U xgboost tensorflow scikit-learn"
echo ""
echo "2. Train XSS model (15-20 min):"
echo "   python xss/train_xss_advanced.py --sample 0.3 --epochs 10"
echo ""
echo "3. Train SQL model (8-12 min):"
echo "   python sql_injection/train_sql_advanced.py --sample 0.2 --epochs 10"
echo ""
echo "4. Restart server (in new terminal):"
echo "   sudo ./run.sh"
echo ""
echo "5. Test in browser:"
echo "   http://127.0.0.1:8000/test-suite"
echo ""
echo "=================================="
