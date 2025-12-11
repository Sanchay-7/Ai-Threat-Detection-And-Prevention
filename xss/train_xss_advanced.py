"""
Advanced XSS Detection using Ensemble Methods + Neural Networks.

Uses:
- TF-IDF + Multiple ML models (Random Forest, SVM, XGBoost)
- Neural Network (LSTM/Dense)
- Voting Ensemble for robust detection
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

# Map column names
XSS_CSV = "dataset/Large-Scale Annotated Dataset for Cross-Site Scripting (XSS) Attack Detection.csv"
COLUMN_MAPPING = {
    'Sentence': 'text',
    'Query': 'text',
    'label': 'label',
    'Label': 'label'
}

def load_xss_dataset(filepath, sample_fraction=1.0):
    """Load XSS dataset with sampling for faster training"""
    print(f"📂 Loading XSS dataset from {filepath}...")
    
    df = pd.read_csv(filepath)
    print(f"   Initial size: {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Rename columns if needed
    df = df.rename(columns=COLUMN_MAPPING)
    
    # Handle missing values
    df = df.dropna(subset=['text', 'label'])
    
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

def build_neural_network(input_dim):
    """Build neural network for XSS detection"""
    print("\n🧠 Building neural network...")
    
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

def train_xss_advanced(csv_path=None, sample_fraction=0.3, epochs=10, batch_size=32):
    """Train advanced XSS detection models"""
    
    # Load data
    csv_file = Path(csv_path) if csv_path else Path(XSS_CSV)
    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_file}")
    
    df = load_xss_dataset(csv_file, sample_fraction=sample_fraction)
    
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
    
    # Convert to dense for neural network
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()
    
    # Scale for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_dense)
    X_test_scaled = scaler.transform(X_test_dense)
    
    # ===== Model 1: Ensemble =====
    print("\n" + "="*60)
    print("MODEL 1: VOTING ENSEMBLE (RF + SVM + XGBoost + LogReg)")
    print("="*60)
    ensemble = build_ensemble_model(X_train_tfidf, y_train)
    
    y_pred_ensemble = ensemble.predict(X_test_tfidf)
    y_pred_proba_ensemble = ensemble.predict_proba(X_test_tfidf)[:, 1]
    
    print("\n📈 Ensemble Results:")
    print(classification_report(y_test, y_pred_ensemble, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_ensemble):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_ensemble):.4f}")
    
    # ===== Model 2: Neural Network =====
    print("\n" + "="*60)
    print("MODEL 2: DEEP NEURAL NETWORK")
    print("="*60)
    nn_model = build_neural_network(X_train_scaled.shape[1])
    
    print("\n🎓 Training neural network...")
    history = nn_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    y_pred_nn_proba = nn_model.predict(X_test_scaled).flatten()
    y_pred_nn = (y_pred_nn_proba > 0.5).astype(int)
    
    print("\n📈 Neural Network Results:")
    print(classification_report(y_test, y_pred_nn, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_nn_proba):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_nn):.4f}")
    
    # ===== Save Models =====
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    # Create model directory
    model_dir = Path(config.XSS_MODEL_PATH).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ensemble
    joblib.dump(vectorizer, model_dir / 'xss_vectorizer.pkl')
    joblib.dump(ensemble, model_dir / 'xss_ensemble.pkl')
    joblib.dump(scaler, model_dir / 'xss_scaler.pkl')
    
    # Save neural network
    nn_model.save(str(model_dir / 'xss_neural_network.h5'))
    
    print(f"✅ Vectorizer saved: {model_dir / 'xss_vectorizer.pkl'}")
    print(f"✅ Ensemble saved: {model_dir / 'xss_ensemble.pkl'}")
    print(f"✅ Scaler saved: {model_dir / 'xss_scaler.pkl'}")
    print(f"✅ Neural Network saved: {model_dir / 'xss_neural_network.h5'}")
    
    # Save metadata
    metadata = {
        'ensemble_f1': float(f1_score(y_test, y_pred_ensemble)),
        'nn_f1': float(f1_score(y_test, y_pred_nn)),
        'ensemble_auc': float(roc_auc_score(y_test, y_pred_proba_ensemble)),
        'nn_auc': float(roc_auc_score(y_test, y_pred_nn_proba)),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    with open(model_dir / 'xss_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved: {model_dir / 'xss_metadata.json'}")
    print(f"\n📊 Summary:\n{json.dumps(metadata, indent=2)}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=XSS_CSV, help='Path to XSS CSV dataset')
    parser.add_argument('--sample', type=float, default=0.3, help='Fraction of data to use (0-1)')
    parser.add_argument('--epochs', type=int, default=10, help='Neural network epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Neural network batch size')
    args = parser.parse_args()
    
    train_xss_advanced(args.dataset, args.sample, args.epochs, args.batch_size)
