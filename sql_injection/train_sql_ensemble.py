"""
Simplified SQL Injection Detection using Ensemble Methods.
Faster training without neural networks - uses scikit-learn only.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

SQLI_CSV = "dataset/SQL_Injection_Detection_Dataset.csv"
COLUMN_MAPPING = {
    'Query': 'text',
    'Label': 'label'
}

def load_sqli_dataset(filepath, sample_fraction=1.0):
    """Load SQL Injection dataset with sampling"""
    print(f"📂 Loading SQL Injection dataset from {filepath}...")
    
    df = pd.read_csv(filepath)
    print(f"   Initial size: {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Rename columns if needed
    df = df.rename(columns=COLUMN_MAPPING)
    
    # Handle missing values
    df = df.dropna(subset=['text', 'label'])
    
    # Convert label to integer (handles string labels like '0', '1')
    try:
        df['label'] = pd.to_numeric(df['label'], errors='coerce').astype('Int64')
    except:
        df['label'] = df['label'].astype(int)
    
    # Clean up labels: only keep 0/1
    df = df[df['label'].isin([0, 1])]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['text'])
    print(f"   After cleaning: {len(df)} rows")
    
    # Sample if needed
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42)
        print(f"   Sampled to: {len(df)} rows")
    
    print(f"   Label distribution:\n{df['label'].value_counts()}")
    return df

def build_ensemble_model(X_train_tfidf, y_train):
    """Build voting ensemble with multiple classifiers"""
    print("\n🔨 Building ensemble model...")
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)),
            ('svm', SVC(kernel='rbf', probability=True, C=1.0, random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, 
                                      tree_method='hist', device='cpu', random_state=42)),
            ('lr', LogisticRegression(max_iter=500, n_jobs=-1, random_state=42))
        ],
        voting='soft'
    )
    
    print("   Training ensemble (RF, SVM, XGBoost, LogReg)...")
    ensemble.fit(X_train_tfidf, y_train)
    return ensemble

def train_sqli_ensemble(csv_path=None, sample_fraction=0.2):
    """Train SQL Injection ensemble model (no neural network)"""
    
    # Load data
    csv_file = Path(csv_path) if csv_path else Path(SQLI_CSV)
    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_file}")
    
    df = load_sqli_dataset(csv_file, sample_fraction=sample_fraction)
    
    # Split data
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    # TF-IDF Vectorization
    print("\n🔤 TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        lowercase=True,
        min_df=2,
        max_features=10000,
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"   TF-IDF matrix shape: {X_train_tfidf.shape}")
    
    # ===== Train Ensemble =====
    print("\n" + "="*60)
    print("TRAINING: VOTING ENSEMBLE (RF + SVM + XGBoost + LogReg)")
    print("="*60)
    ensemble = build_ensemble_model(X_train_tfidf, y_train)
    
    y_pred_ensemble = ensemble.predict(X_test_tfidf)
    y_pred_proba_ensemble = ensemble.predict_proba(X_test_tfidf)[:, 1]
    
    print("\n📈 Ensemble Results:")
    print(classification_report(y_test, y_pred_ensemble, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_ensemble):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_ensemble):.4f}")
    
    # ===== Save Models =====
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    # Create model directory
    model_dir = Path(config.SQL_MODEL_PATH).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    joblib.dump(vectorizer, model_dir / 'sql_vectorizer.pkl')
    joblib.dump(ensemble, model_dir / 'sql_ensemble.pkl')
    
    print(f"✅ Vectorizer saved: {model_dir / 'sql_vectorizer.pkl'}")
    print(f"✅ Ensemble saved: {model_dir / 'sql_ensemble.pkl'}")
    
    # Save metadata
    metadata = {
        'ensemble_f1': float(f1_score(y_test, y_pred_ensemble)),
        'ensemble_auc': float(roc_auc_score(y_test, y_pred_proba_ensemble)),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'model_type': 'ensemble_only'
    }
    
    with open(model_dir / 'sql_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved: {model_dir / 'sql_metadata.json'}")
    print(f"\n📊 Summary:\n{json.dumps(metadata, indent=2)}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=SQLI_CSV, help='Path to SQLi CSV dataset')
    parser.add_argument('--sample', type=float, default=0.2, help='Fraction of data to use (0-1)')
    args = parser.parse_args()
    
    train_sqli_ensemble(args.dataset, args.sample)
