#!/usr/bin/env bash
set -e

echo "🚀 AI DDoS Shield - Advanced Model Training"
echo "==========================================="
echo ""
echo "This will train advanced ensemble + neural network models"
echo "using real datasets from the dataset/ directory"
echo ""

cd "$(dirname "$0")"
source .venv/bin/activate

echo "📦 Checking dependencies..."
pip install -q xgboost tensorflow -U

echo ""
echo "🔨 Training Advanced XSS Detection Model..."
echo "⏳ This may take 10-20 minutes depending on sample size..."
python xss/train_xss_advanced.py --sample 0.3 --epochs 10 --batch-size 32

echo ""
echo "================================================"
echo ""
echo "🔨 Training Advanced SQL Injection Detection Model..."
echo "⏳ This may take 5-10 minutes depending on sample size..."
python sql_injection/train_sql_advanced.py --sample 0.2 --epochs 10 --batch-size 32

echo ""
echo "================================================"
echo "✅ Training Complete!"
echo ""
echo "📊 Models Created:"
echo "   ✓ xss/xss_vectorizer.pkl - TF-IDF vectorizer"
echo "   ✓ xss/xss_ensemble.pkl - Ensemble classifier"
echo "   ✓ xss/xss_neural_network.h5 - Neural network"
echo "   ✓ xss/xss_metadata.json - Performance metrics"
echo ""
echo "   ✓ sql_injection/sql_vectorizer.pkl - TF-IDF vectorizer"
echo "   ✓ sql_injection/sql_ensemble.pkl - Ensemble classifier"
echo "   ✓ sql_injection/sql_neural_network.h5 - Neural network"
echo "   ✓ sql_injection/sql_metadata.json - Performance metrics"
echo ""
echo "🎯 To use advanced models, update app.py to use:"
echo "   from xss.inference_xss_advanced import predict as predict_xss"
echo "   from sql_injection.inference_sql_advanced import predict as predict_sql"
echo ""
