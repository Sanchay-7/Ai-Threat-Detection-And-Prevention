"""
Fast SQL Injection Detection Training.
Uses only RF + XGBoost (2 classifiers) for speed.
Adds CLI flags to control sample size and feature budget for quicker runs.
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
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

SQLI_CSV = "dataset/SQL_Injection_Detection_Dataset.csv"

def load_sqli_dataset(filepath, sample_fraction=1.0):
    """Load SQL Injection dataset with sampling"""
    print(f"📂 Loading SQL Injection dataset...")
    
    df = pd.read_csv(filepath, usecols=['Query', 'Label'])
    print(f"   Initial size: {len(df)} rows")
    
    # Rename and clean
    df = df.rename(columns={'Query': 'text', 'Label': 'label'})
    df = df.dropna(subset=['text', 'label'])
    
    # Convert label to int (handles string labels)
    df['label'] = pd.to_numeric(df['label'], errors='coerce').astype('Int64')
    df = df[df['label'].isin([0, 1])]
    
    # Remove duplicates  
    df = df.drop_duplicates(subset=['text'])
    print(f"   After cleaning: {len(df)} rows")
    
    # Sample
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42)
        print(f"   Using sample: {len(df)} rows")
    
    print(f"   Label distribution: {df['label'].value_counts().to_dict()}")
    return df

def train(sample_fraction: float = 0.15, max_features: int = 3000, n_estimators: int = 40):
    """Train SQL Injection fast model"""

    csv_file = Path(SQLI_CSV)
    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_file}")

    df = load_sqli_dataset(csv_file, sample_fraction=sample_fraction)

    print("\n📊 Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    print("🔤 TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        lowercase=True,
        min_df=2,
        max_features=max_features,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"   TF-IDF matrix: {X_train_tfidf.shape}")

    print("\n" + "="*50)
    print("TRAINING: RF + XGBoost Ensemble")
    print("="*50)

    ensemble = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=n_estimators, max_depth=12, n_jobs=-1, random_state=42)),
            ('xgb', xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                learning_rate=0.1,
                tree_method='hist',
                device='cpu',
                random_state=42
            )),
        ],
        voting='soft'
    )

    print("Training ensemble...")
    ensemble.fit(X_train_tfidf, y_train)

    y_pred = ensemble.predict(X_test_tfidf)
    y_pred_proba = ensemble.predict_proba(X_test_tfidf)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("\n📈 Results:")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC-AUC:  {auc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    print("\n" + "="*50)
    print("SAVING MODELS")
    print("="*50)

    model_dir = Path(config.SQL_MODEL_PATH).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, model_dir / 'sql_vectorizer.pkl')
    joblib.dump(ensemble, model_dir / 'sql_ensemble.pkl')

    metadata = {
        'f1_score': float(f1),
        'auc_score': float(auc),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'model_type': 'rf_xgb_ensemble',
        'sample_fraction': sample_fraction,
        'max_features': max_features,
        'n_estimators': n_estimators,
    }

    with open(model_dir / 'sql_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("✅ Vectorizer saved")
    print("✅ Ensemble saved")
    print("✅ Metadata saved")
    print(f"\n📊 Summary:\n{json.dumps(metadata, indent=2)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train fast SQLi ensemble")
    parser.add_argument('--sample', type=float, default=0.15, help='Fraction of data to sample (0-1]')
    parser.add_argument('--max-features', type=int, default=3000, help='TF-IDF max_features')
    parser.add_argument('--n-estimators', type=int, default=40, help='Trees for RF/XGB')
    args = parser.parse_args()

    train(sample_fraction=args.sample, max_features=args.max_features, n_estimators=args.n_estimators)
