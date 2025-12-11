"""Train a SQL injection classifier (TF-IDF + Logistic Regression).

Can train on large JSONL dataset or CSV. If neither provided, uses synthetic bootstrap data.
Expected format: JSONL with 'payload' and 'label' fields, or CSV with 'text' and 'label' columns.
"""
from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


FALLBACK_SAMPLES = [
    ("' OR 1=1 --", 1),
    ("admin' #", 1),
    ("union select username, password from users", 1),
    ("select * from users where id=1", 0),
    ("/products?id=5", 0),
    ("normal text", 0),
]


def load_dataset(path: Path | None) -> pd.DataFrame:
    """Load dataset from JSONL, CSV, or use fallback synthetic data"""
    if path and path.exists():
        if path.suffix == '.jsonl':
            # Load JSONL format
            data = []
            with open(path, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    data.append({'text': item['payload'], 'label': item['label']})
            return pd.DataFrame(data)
        else:
            # Load CSV format
            df = pd.read_csv(path)
            if "text" not in df.columns or "label" not in df.columns:
                raise ValueError("Dataset must have 'text' and 'label' columns")
            return df
    return pd.DataFrame(FALLBACK_SAMPLES, columns=["text", "label"])


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True, min_df=1, max_features=5000)),
        ("clf", LogisticRegression(max_iter=200, n_jobs=-1)),
    ])


def train(dataset_path: str | None = None, model_path: str | None = None):
    data_path = Path(dataset_path) if dataset_path else None
    out_path = Path(model_path) if model_path else Path(config.SQL_MODEL_PATH)

    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"] if df["label"].nunique() > 1 else None)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    if len(X_test) > 0:
        print(classification_report(y_test, pipe.predict(X_test)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    print(f"Saved SQLi model to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to CSV with text,label", default=None)
    parser.add_argument("--output", help="Where to store model", default=None)
    args = parser.parse_args()
    train(args.dataset, args.output)
