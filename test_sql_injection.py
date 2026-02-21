#!/usr/bin/env python3
"""
SQL Injection Testing Script
Tests the detector with payloads from the training dataset.
FOR AUTHORIZED TESTING ONLY - Use on your own systems only.
"""
import pandas as pd
import requests
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Configuration
API_URL = "http://127.0.0.1:8000/test"
DATASET_PATH = "dataset/SQL_Injection_Detection_Dataset.csv"
BATCH_SIZE = 50
DELAY_BETWEEN_REQUESTS = 0.05  # 50ms between requests

class SQLInjectionTester:
    def __init__(self, api_url: str, dataset_path: str):
        self.api_url = api_url
        self.dataset_path = dataset_path
        self.results = {
            "detected": 0,
            "missed": 0,
            "errors": 0,
            "total": 0,
            "payloads": []
        }

    def load_dataset(self) -> pd.DataFrame:
        """Load SQL injection dataset"""
        if not Path(self.dataset_path).exists():
            print(f"❌ Dataset not found at {self.dataset_path}")
            sys.exit(1)
        
        try:
            df = pd.read_csv(self.dataset_path)
            print(f"✅ Loaded {len(df)} SQL injection payloads from dataset")
            return df
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            sys.exit(1)

    def get_attack_payloads(self, df: pd.DataFrame) -> List[str]:
        """Extract attack payloads (Label=1) from dataset"""
        # Convert Label to numeric, coerce errors to NaN
        df_copy = df.copy()
        df_copy['Label'] = pd.to_numeric(df_copy['Label'], errors='coerce')
        attacks = df_copy[df_copy['Label'] == 1]['Query'].dropna().astype(str).unique()
        # Remove any placeholder strings
        attacks = [p for p in attacks if p and p.strip() and p != 'Query']
        return list(attacks)

    def get_benign_payloads(self, df: pd.DataFrame) -> List[str]:
        """Extract benign payloads (Label=0) from dataset"""
        df_copy = df.copy()
        df_copy['Label'] = pd.to_numeric(df_copy['Label'], errors='coerce')
        benign = df_copy[df_copy['Label'] == 0]['Query'].dropna().astype(str).unique()
        benign = [p for p in benign if p and p.strip() and p != 'Query']
        return list(benign)

    def test_payload(self, payload: str) -> Tuple[bool, dict]:
        """Test a single payload against the detector"""
        try:
            response = requests.post(self.api_url, data=payload, timeout=5)
            result = response.json()
            detected = result.get('sql', {}).get('decision', False)
            return True, {
                'payload': payload[:80],
                'detected': detected,
                'score': result.get('sql', {}).get('score', 0),
                'reason': result.get('sql', {}).get('reason', 'unknown')
            }
        except Exception as e:
            return False, {'payload': payload[:80], 'error': str(e)}

    def run_attack_tests(self, payloads: List[str], label: str = "Attack Payloads") -> Dict:
        """Test attack payloads - should be detected"""
        print(f"\n{'='*70}")
        print(f"🔴 Testing {len(payloads)} {label}")
        print(f"{'='*70}")
        
        stats = {"detected": 0, "missed": 0, "errors": 0}
        missed_payloads = []
        
        for i, payload in enumerate(payloads):
            success, result = self.test_payload(payload)
            
            if not success:
                stats["errors"] += 1
                status = "❌ ERROR"
            elif result.get('detected'):
                stats["detected"] += 1
                status = "✅ DETECTED"
            else:
                stats["missed"] += 1
                status = "⚠️  MISSED"
                missed_payloads.append(result)
            
            # Print progress every 10 payloads
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(payloads)} | Detected: {stats['detected']} | Missed: {stats['missed']}")
            
            # Add to results
            self.results["payloads"].append({
                "type": "attack",
                "status": status,
                **result
            })
            
            time.sleep(DELAY_BETWEEN_REQUESTS)
        
        # Print summary
        print(f"\n📊 Results:")
        print(f"  ✅ Detected: {stats['detected']}/{len(payloads)} ({100*stats['detected']/len(payloads):.1f}%)")
        print(f"  ⚠️  Missed: {stats['missed']}/{len(payloads)} ({100*stats['missed']/len(payloads):.1f}%)")
        print(f"  ❌ Errors: {stats['errors']}/{len(payloads)}")
        
        if missed_payloads and len(missed_payloads) <= 5:
            print(f"\n⚠️  Missed payloads:")
            for p in missed_payloads:
                print(f"    • {p['payload'][:70]}...")
        
        self.results["detected"] += stats["detected"]
        self.results["missed"] += stats["missed"]
        self.results["errors"] += stats["errors"]
        
        return stats

    def run_benign_tests(self, payloads: List[str]) -> Dict:
        """Test benign payloads - should NOT be detected"""
        print(f"\n{'='*70}")
        print(f"🟢 Testing {len(payloads)} Benign Payloads (False Positives)")
        print(f"{'='*70}")
        
        stats = {"correct": 0, "false_positive": 0, "errors": 0}
        false_positives = []
        
        for i, payload in enumerate(payloads):
            success, result = self.test_payload(payload)
            
            if not success:
                stats["errors"] += 1
                status = "❌ ERROR"
            elif not result.get('detected'):
                stats["correct"] += 1
                status = "✅ CORRECT"
            else:
                stats["false_positive"] += 1
                status = "⚠️  FALSE POSITIVE"
                false_positives.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(payloads)} | Correct: {stats['correct']} | FP: {stats['false_positive']}")
            
            self.results["payloads"].append({
                "type": "benign",
                "status": status,
                **result
            })
            
            time.sleep(DELAY_BETWEEN_REQUESTS)
        
        print(f"\n📊 Results:")
        print(f"  ✅ Correct (not flagged): {stats['correct']}/{len(payloads)} ({100*stats['correct']/len(payloads):.1f}%)")
        print(f"  ⚠️  False Positives: {stats['false_positive']}/{len(payloads)} ({100*stats['false_positive']/len(payloads):.1f}%)")
        print(f"  ❌ Errors: {stats['errors']}/{len(payloads)}")
        
        if false_positives and len(false_positives) <= 5:
            print(f"\n⚠️  False positive payloads:")
            for p in false_positives:
                print(f"    • {p['payload'][:70]}... (score: {p.get('score', 0):.3f})")
        
        return stats

    def print_final_summary(self):
        """Print final test summary"""
        print(f"\n{'='*70}")
        print(f"📈 FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"  Total payloads tested: {self.results['total']}")
        print(f"  Attacks detected: {self.results['detected']}")
        print(f"  Attacks missed: {self.results['missed']}")
        print(f"  Errors: {self.results['errors']}")
        
        if self.results["detected"] + self.results["missed"] > 0:
            detection_rate = 100 * self.results['detected'] / (self.results['detected'] + self.results['missed'])
            print(f"\n  🎯 Detection Rate: {detection_rate:.1f}%")

    def run(self, test_sample_size: int = None):
        """Run complete test suite"""
        print(f"🚀 SQL Injection Detector Testing Script")
        print(f"Target: {self.api_url}")
        print(f"Dataset: {self.dataset_path}\n")
        
        # Load dataset
        df = self.load_dataset()
        
        # Get payloads
        attack_payloads = self.get_attack_payloads(df)
        benign_payloads = self.get_benign_payloads(df)
        
        # Limit sample size if specified
        if test_sample_size:
            attack_payloads = attack_payloads[:test_sample_size]
            benign_payloads = benign_payloads[:test_sample_size]
            print(f"⚙️  Limiting to {test_sample_size} payloads per category for faster testing\n")
        
        self.results["total"] = len(attack_payloads) + len(benign_payloads)
        
        if len(attack_payloads) == 0:
            print("❌ No attack payloads found in dataset")
            return None
        if len(benign_payloads) == 0:
            print("❌ No benign payloads found in dataset")
            return None
        
        # Test attacks
        attack_stats = self.run_attack_tests(attack_payloads)
        
        # Test benign
        benign_stats = self.run_benign_tests(benign_payloads)
        
        # Print summary
        self.print_final_summary()
        
        return {
            "attacks": attack_stats,
            "benign": benign_stats,
            "total": self.results
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SQL Injection Detection Testing Script"
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/test",
        help="API endpoint URL (default: http://127.0.0.1:8000/test)"
    )
    parser.add_argument(
        "--dataset",
        default="dataset/SQL_Injection_Detection_Dataset.csv",
        help="Path to SQL injection dataset CSV"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Test only N payloads per category (for faster testing)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Delay between requests in seconds (default: 0.05)"
    )
    
    args = parser.parse_args()
    
    # Update globals
    global DELAY_BETWEEN_REQUESTS
    DELAY_BETWEEN_REQUESTS = args.delay
    
    # Create tester
    tester = SQLInjectionTester(args.api_url, args.dataset)
    
    try:
        results = tester.run(test_sample_size=args.sample)
        return 0
    except KeyboardInterrupt:
        print("\n\n⏸️  Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
