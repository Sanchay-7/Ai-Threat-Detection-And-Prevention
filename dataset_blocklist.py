"""
Build blocklists from training datasets.
Treats known attack payloads from datasets as hard-blocks.
"""
import hashlib
from pathlib import Path
import pandas as pd
from typing import Set

# Cache for blocklists
_XSS_BLOCKLIST: Set[str] = set()
_SQL_BLOCKLIST: Set[str] = set()
_BLOCKLIST_LOADED = False

def _hash_payload(payload: str) -> str:
    """Hash a payload for fast lookup."""
    return hashlib.md5(payload.encode('utf-8')).hexdigest()

def load_blocklists():
    """Load attack samples from datasets into memory blocklists."""
    global _XSS_BLOCKLIST, _SQL_BLOCKLIST, _BLOCKLIST_LOADED
    
    if _BLOCKLIST_LOADED:
        return
    
    # Load XSS attacks
    xss_path = Path("dataset/Large-Scale Annotated Dataset for Cross-Site Scripting (XSS) Attack Detection.csv")
    if xss_path.exists():
        try:
            df = pd.read_csv(xss_path)
            # Filter for attack samples (label=1)
            attacks = df[df.get('Label', df.get('label', pd.Series())) == 1]
            # Get the payload column (could be 'Query' or 'Sentence')
            payload_col = 'Query' if 'Query' in df.columns else 'Sentence'
            for payload in attacks[payload_col].dropna().unique():
                _XSS_BLOCKLIST.add(_hash_payload(str(payload)))
            print(f"✅ Loaded {len(_XSS_BLOCKLIST)} XSS attack hashes")
        except Exception as e:
            print(f"⚠️ Failed to load XSS blocklist: {e}")
    
    # Load SQL Injection attacks
    sql_path = Path("dataset/SQL_Injection_Detection_Dataset.csv")
    if sql_path.exists():
        try:
            df = pd.read_csv(sql_path)
            # Filter for attack samples (Label=1 or Label=1.0)
            attacks = df[df['Label'].isin([1, 1.0, '1', True])]
            # Get the payload column (should be 'Query')
            if 'Query' in df.columns:
                for payload in attacks['Query'].dropna().astype(str).unique():
                    if payload and payload.strip() and payload != 'Query':  # Skip empty payloads
                        _SQL_BLOCKLIST.add(_hash_payload(str(payload)))
                print(f"✅ Loaded {len(_SQL_BLOCKLIST)} SQL injection attack hashes")
            else:
                print("⚠️ Could not find Query column in SQL dataset")
        except Exception as e:
            print(f"⚠️ Failed to load SQL blocklist: {e}")
    
    _BLOCKLIST_LOADED = True

def check_xss_blocklist(payload: str) -> bool:
    """Check if payload is in XSS blocklist."""
    if not _BLOCKLIST_LOADED:
        load_blocklists()
    return _hash_payload(payload) in _XSS_BLOCKLIST

def check_sql_blocklist(payload: str) -> bool:
    """Check if payload is in SQL blocklist."""
    if not _BLOCKLIST_LOADED:
        load_blocklists()
    return _hash_payload(payload) in _SQL_BLOCKLIST

if __name__ == "__main__":
    load_blocklists()
    print(f"\nTotal XSS attacks in blocklist: {len(_XSS_BLOCKLIST)}")
    print(f"Total SQL attacks in blocklist: {len(_SQL_BLOCKLIST)}")
