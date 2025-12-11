import os
import pandas as pd
import argparse
import config
from detector import HybridDetector

def main():
    if not os.path.exists(config.GENERATED_DATASET_PATH):
        print(f"Error: Dataset not found at '{config.GENERATED_DATASET_PATH}'.")
        print("Please run 'python3 generate_dataset.py' first to create it.")
        return

    print("Initializing detector and starting training process...")
    
    # Ensure model directory exists
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    hd = HybridDetector()
    
    print("\n--- Training Supervised Model ---")
    hd.train_supervised(config.GENERATED_DATASET_PATH)
    
    print("\n--- Training Anomaly Model ---")
    hd.train_anomaly(config.GENERATED_DATASET_PATH)

    print("\n--- Training MLP Model ---")
    hd.train_mlp(config.GENERATED_DATASET_PATH)

    print("\n--- Training Autoencoder Model ---")
    hd.train_autoencoder(config.GENERATED_DATASET_PATH)
    
    print("\n✅ Training finished successfully.")
    print(f"Models saved to '{config.SUPERVISED_MODEL_PATH}' and '{config.ANOMALY_MODEL_PATH}'.")

if __name__ == "__main__":
    main()
