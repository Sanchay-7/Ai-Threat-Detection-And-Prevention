"""
Advanced SQL Injection Detection using Ensemble Methods + Neural Networks.

Uses:
- TF-IDF + Multiple ML models (Random Forest, SVM, XGBoost)
- Neural Network (Dense)
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
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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
SQLI_CSV = "dataset/SQL_Injection_Detection_Dataset.csv"
COLUMN_MAPPING = {
    'Query': 'text',
    'Label': 'label'
}

def load_sqli_dataset(filepath, sample_fraction=1.0):
    """Load SQLi dataset with sampling for faster training"""
    print(f"📂 Loading SQL Injection dataset from {filepath}...")
    
    df = pd.read_csv(filepath)
    print(f"   Initial size: {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Rename columns if needed
    df = df.rename(columns=COLUMN_MAPPING)
    
    # Handle missing values
    df = df.dropna(subset=['text', 'label'])
    
    # Clean and convert labels to binary (0 or 1)
    # Handle string versions and numeric versions
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label'] = df['label'].map(lambda x: 1 if x in ['1', '1.0', 'true', 'sqli'] else (0 if x in ['0', '0.0', 'false', 'benign'] else None))
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print(f"   After cleaning: {len(df)} rows")
    
    # Sample if needed
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42)
        print(f"   Sampled to: {len(df)} rows")
    
    print(f"   Label distribution:\n{df['label'].value_counts()}")
    return df


def detect_gpu():
    """Return True if TensorFlow sees a GPU; enable memory growth when present."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    return bool(gpus)


def build_ensemble_model(X_train_tfidf, y_train, use_gpu=False):
    """Build voting ensemble with multiple classifiers"""
    print("\n🔨 Building ensemble model...")

    linear_svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, random_state=42),
        method='sigmoid',
        cv=3
    )

    xgb_params = dict(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.1,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        subsample=0.9,
        colsample_bytree=0.9
    )
    if use_gpu:
        xgb_params.update(tree_method='gpu_hist', predictor='gpu_predictor')

    ensemble = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)),
            ('svm', linear_svm),
            ('xgb', xgb.XGBClassifier(**xgb_params)),
            ('lr', LogisticRegression(max_iter=500, n_jobs=-1, random_state=42))
        ],
        voting='soft'
    )
    
    print(f"   Training ensemble (RF, Linear SVM, XGBoost{' GPU' if use_gpu else ''}, LogReg)...")
    ensemble.fit(X_train_tfidf, y_train)
    return ensemble

def build_neural_network(input_dim):
    """Build neural network for SQLi detection"""
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

def train_sqli_advanced(csv_path=None, sample_fraction=0.2, epochs=10, batch_size=32,
                       max_features=10000, skip_nn=False, nn_sample=1.0):
    """Train advanced SQL Injection detection models.

    skip_nn: set True to avoid dense NN step (helpful for very large datasets).
    nn_sample: fraction of train/test data to use for NN (keeps ensemble on full data).
    """
    
    # Load data
    csv_file = Path(csv_path) if csv_path else Path(SQLI_CSV)
    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_file}")
    
    df = load_sqli_dataset(csv_file, sample_fraction=sample_fraction)
    
    has_gpu = detect_gpu()
    print(f"\n🖥️ GPU available: {has_gpu}")
    
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
        max_features=max_features,
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"   TF-IDF matrix shape: {X_train_tfidf.shape}")
    
    # ===== Model 1: Ensemble =====
    print("\n" + "="*60)
    print("MODEL 1: VOTING ENSEMBLE (RF + SVM + XGBoost + LogReg)")
    print("="*60)
    ensemble = build_ensemble_model(X_train_tfidf, y_train, use_gpu=has_gpu)
    
    y_pred_ensemble = ensemble.predict(X_test_tfidf)
    y_pred_proba_ensemble = ensemble.predict_proba(X_test_tfidf)[:, 1]
    
    print("\n📈 Ensemble Results:")
    print(classification_report(y_test, y_pred_ensemble, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_ensemble):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_ensemble):.4f}")
    
    # ===== Model 2: Neural Network (optional) =====
    if skip_nn:
        print("\n⏭️ Skipping neural network (skip_nn=True).")
        nn_model = None
        y_pred_nn = y_pred_nn_proba = None
    else:
        nn_sample = float(nn_sample)
        nn_sample = 1.0 if nn_sample > 1.0 else nn_sample
        rng = np.random.RandomState(42)

        if nn_sample < 1.0:
            train_count = int(len(y_train) * nn_sample)
            test_count = int(len(y_test) * nn_sample)
            train_idx = rng.choice(len(y_train), size=max(1, train_count), replace=False)
            test_idx = rng.choice(len(y_test), size=max(1, test_count), replace=False)
            X_train_tfidf_nn = X_train_tfidf[train_idx]
            y_train_nn = y_train.iloc[train_idx]
            X_test_tfidf_nn = X_test_tfidf[test_idx]
            y_test_nn = y_test.iloc[test_idx]
            print(f"\n🧪 NN using subset: train {len(y_train_nn)}, test {len(y_test_nn)} (nn_sample={nn_sample})")
        else:
            X_train_tfidf_nn, y_train_nn = X_train_tfidf, y_train
            X_test_tfidf_nn, y_test_nn = X_test_tfidf, y_test
            print("\n🧠 NN using full train/test sets.")

        # Convert to dense for neural network
        X_train_dense = X_train_tfidf_nn.toarray().astype(np.float32)
        X_test_dense = X_test_tfidf_nn.toarray().astype(np.float32)

        # Scale for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_dense)
        X_test_scaled = scaler.transform(X_test_dense)

        print("\n" + "="*60)
        print("MODEL 2: DEEP NEURAL NETWORK")
        print("="*60)
        nn_model = build_neural_network(X_train_scaled.shape[1])
        
        print("\n🎓 Training neural network...")
        history = nn_model.fit(
            X_train_scaled, y_train_nn,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        y_pred_nn_proba = nn_model.predict(X_test_scaled).flatten()
        y_pred_nn = (y_pred_nn_proba > 0.5).astype(int)
        
        print("\n📈 Neural Network Results:")
        print(classification_report(y_test_nn, y_pred_nn, digits=4))
        print(f"ROC-AUC: {roc_auc_score(y_test_nn, y_pred_nn_proba):.4f}")
        print(f"F1-Score: {f1_score(y_test_nn, y_pred_nn):.4f}")
    
    # ===== Save Models =====
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    # Create model directory
    model_dir = Path(config.SQL_MODEL_PATH).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ensemble
    joblib.dump(vectorizer, model_dir / 'sql_vectorizer.pkl')
    joblib.dump(ensemble, model_dir / 'sql_ensemble.pkl')
    
    if not skip_nn and nn_model is not None:
        joblib.dump(scaler, model_dir / 'sql_scaler.pkl')
        nn_model.save(str(model_dir / 'sql_neural_network.h5'))
        print(f"✅ Scaler saved: {model_dir / 'sql_scaler.pkl'}")
        print(f"✅ Neural Network saved: {model_dir / 'sql_neural_network.h5'}")
    
    print(f"✅ Vectorizer saved: {model_dir / 'sql_vectorizer.pkl'}")
    print(f"✅ Ensemble saved: {model_dir / 'sql_ensemble.pkl'}")
    
    # Save metadata
    metadata = {
        'ensemble_f1': float(f1_score(y_test, y_pred_ensemble)),
        'ensemble_auc': float(roc_auc_score(y_test, y_pred_proba_ensemble)),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'max_features': max_features,
        'used_gpu': has_gpu,
        'skip_nn': skip_nn,
        'nn_sample': nn_sample
    }
    
    if not skip_nn and y_pred_nn is not None:
        metadata.update({
            'nn_f1': float(f1_score(y_test_nn, y_pred_nn)),
            'nn_auc': float(roc_auc_score(y_test_nn, y_pred_nn_proba)),
            'nn_train_samples': len(y_train_nn),
            'nn_test_samples': len(y_test_nn)
        })
    
    with open(model_dir / 'sql_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved: {model_dir / 'sql_metadata.json'}")
    print(f"\n📊 Summary:\n{json.dumps(metadata, indent=2)}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=SQLI_CSV, help='Path to SQLi CSV dataset')
    parser.add_argument('--sample', type=float, default=0.2, help='Fraction of data to use (0-1)')
    parser.add_argument('--epochs', type=int, default=10, help='Neural network epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Neural network batch size')
    parser.add_argument('--max-features', type=int, default=10000, help='TF-IDF max features')
    parser.add_argument('--skip-nn', action='store_true', help='Skip neural network training to save time/memory')
    parser.add_argument('--nn-sample', type=float, default=1.0, help='Fraction of train/test used for neural net (0-1)')
    args = parser.parse_args()
    
    train_sqli_advanced(
        args.dataset,
        args.sample,
        args.epochs,
        args.batch_size,
        max_features=args.max_features,
        skip_nn=args.skip_nn,
        nn_sample=args.nn_sample
    )
