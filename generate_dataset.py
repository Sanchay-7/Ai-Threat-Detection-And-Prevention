import pandas as pd
import numpy as np
import os
import config

def generate_traffic_data(n_samples=50000):
    """
    Generates a synthetic dataset of HTTP traffic features for training.
    Features are designed to be extractable from a web server context.
    """
    print(f"Generating a synthetic dataset with {n_samples} samples...")
    
    # --- Normal Traffic ---
    n_normal = int(n_samples * 0.95)
    normal_data = {
        'req_rate': np.random.uniform(0.1, 10, n_normal), # Requests per second from this IP
        'unique_paths_rate': np.random.uniform(0.1, 5, n_normal), # Unique URLs per second
        'ip_entropy': np.random.uniform(0.5, 4.0, n_normal), # A measure of randomness in recent source IPs
        'payload_size': np.random.randint(50, 2048, n_normal),
        'label': 0 # 0 for 'normal'
    }
    normal_df = pd.DataFrame(normal_data)

    # --- DDoS Traffic ---
    n_ddos = n_samples - n_normal
    ddos_data = {
        'req_rate': np.random.uniform(50, 500, n_ddos), # Very high request rate
        'unique_paths_rate': np.random.uniform(0.01, 1, n_ddos), # Often hit the same URL
        'ip_entropy': np.random.uniform(0.0, 0.5, n_ddos), # Low entropy if it's a single attacker
        'payload_size': np.random.randint(10, 500, n_ddos), # Often smaller, simpler requests
        'label': 1 # 1 for 'ddos'
    }
    ddos_df = pd.DataFrame(ddos_data)

    # Combine and shuffle
    df = pd.concat([normal_df, ddos_df]).sample(frac=1).reset_index(drop=True)
    
    # Ensure directory exists
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    
    # Save to CSV
    df.to_csv(config.GENERATED_DATASET_PATH, index=False)
    print(f"Successfully generated and saved dataset to '{config.GENERATED_DATASET_PATH}'")
    print("\nDataset preview:")
    print(df.head())
    print("\nClass distribution:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    generate_traffic_data()
