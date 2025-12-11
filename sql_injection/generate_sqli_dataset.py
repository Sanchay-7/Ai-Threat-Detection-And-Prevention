"""
Generate large synthetic dataset for SQL Injection detection training.
Creates both safe payloads and SQLi attack patterns.
"""
import os
import sys
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# SQL Injection attack patterns
SQLI_VECTORS = [
    # OR-based attacks
    "' OR '1'='1",
    "' OR 1=1 --",
    "' OR 1=1 /*",
    "' OR 'x'='x",
    "admin' OR '1'='1' --",
    "' OR 'a'='a",
    "1' OR '1' = '1",
    
    # UNION-based attacks
    "' UNION SELECT NULL --",
    "' UNION SELECT NULL,NULL --",
    "' UNION SELECT NULL,NULL,NULL --",
    "' UNION SELECT username,password FROM users --",
    "1 UNION SELECT NULL,table_name FROM information_schema.tables --",
    
    # Time-based blind
    "' AND SLEEP(5) --",
    "'; WAITFOR DELAY '00:00:05' --",
    "' AND 1=1 AND SLEEP(5) --",
    "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a) --",
    
    # Boolean-based blind
    "' AND 1=1 --",
    "' AND 1=2 --",
    "' AND '1'='1",
    "' AND '1'='2",
    "admin' AND '1'='1",
    
    # Stacked queries
    "'; DROP TABLE users; --",
    "'; DELETE FROM users; --",
    "'; UPDATE users SET admin=1; --",
    "'; INSERT INTO users VALUES('hacker','pass',1); --",
    
    # Comment variations
    "' OR 1=1 #",
    "' OR 1=1 --",
    "' OR 1=1 /*",
    "' OR 1=1 ;%00",
    
    # Encoding bypass
    "' UNION SELECT CONCAT(user,'::',password) FROM admin --",
    "' AND SUBSTRING(version(),1,1)='5' --",
    "' OR CHAR(65)=CHAR(65) --",
    
    # Advanced attacks
    "1' AND (SELECT COUNT(*) FROM information_schema.tables)>0 --",
    "' OR EXISTS(SELECT * FROM users WHERE username='admin') --",
    "'; SELECT @@version; --",
    "' UNION ALL SELECT @@version --",
    
    # Column enumeration
    "' ORDER BY 1 --",
    "' ORDER BY 2 --",
    "' ORDER BY 100 --",
    
    # Nested queries
    "1' AND 1=(SELECT COUNT(*) FROM users WHERE '1'='1') --",
    "' AND (SELECT COUNT(*) FROM users)>0 --",
    
    # NoSQL injection patterns
    "{'$ne':null}",
    "{'$regex':'.*'}",
    "{'$gt':''}",
    
    # LDAP injection patterns
    "*)(uid=*))(|(uid=*",
    "admin*",
    
    # Command injection with SQL context
    "'; exec xp_cmdshell('dir'); --",
    "'; EXECUTE sp_executesql; --",
]

# Safe payloads (non-malicious database queries/inputs)
SAFE_PAYLOADS = [
    'john@example.com',
    'Jane Doe',
    'password123',
    'user_account_5',
    'SELECT * FROM products',
    'UPDATE profile SET bio = ...',
    'INSERT INTO logs VALUES(...)',
    '2024-01-15',
    'product_id_12345',
    'John Smith',
    'order@example.com',
    'city: New York',
    'country: United States',
    'Product Name: Laptop',
    'Price: 999.99',
    'Quantity: 10',
    'Category: Electronics',
    'Description: High performance device',
    'Rating: 4.5 stars',
    'Status: Active',
    'Created: 2024-01-01',
    'Modified: 2024-01-15',
    'Username: john_doe',
    'Email: user@domain.com',
    'Phone: +1-555-0123',
    'Address: 123 Main St',
    'ZIP: 12345',
    'Country Code: US',
    'Language: English',
    'Currency: USD',
    'Transaction ID: TXN123456',
    'Invoice Number: INV-2024-001',
    'Tracking: TRACK123456',
    'Reference: REF-12345',
    'Batch: BATCH001',
    'Version: 1.0.0',
    'Build: 12345',
    'Timestamp: 2024-01-15T10:30:00Z',
    'Hash: a1b2c3d4e5f6',
    'Token: abc123def456',
    'Session: sess_12345',
]

def generate_sqli_dataset(num_samples=50000):
    """Generate large SQLi dataset with balanced labels"""
    dataset = []
    
    # Generate SQLi samples (50% of dataset)
    num_sqli = num_samples // 2
    for i in range(num_sqli):
        payload = random.choice(SQLI_VECTORS)
        # Add variations
        if random.random() < 0.3:
            payload = payload.replace('--', '#')
        if random.random() < 0.2:
            payload = '  ' + payload + '  '  # Add whitespace
        if random.random() < 0.1:
            payload = payload.lower()
        dataset.append({'payload': payload, 'label': 1})
    
    # Generate safe samples (50% of dataset)
    num_safe = num_samples - num_sqli
    for i in range(num_safe):
        payload = random.choice(SAFE_PAYLOADS)
        # Add realistic variations
        if random.random() < 0.3:
            payload = f"{payload} {random.randint(1, 999)}"
        if random.random() < 0.2:
            payload = f"[{payload}]"
        dataset.append({'payload': payload, 'label': 0})
    
    # Shuffle
    random.shuffle(dataset)
    
    return dataset

def save_dataset(dataset, filepath):
    """Save dataset to JSONL format"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
    print(f"✅ Saved {len(dataset)} samples to {filepath}")

if __name__ == '__main__':
    print("🔨 Generating large SQL Injection dataset...")
    sqli_data = generate_sqli_dataset(50000)
    save_dataset(sqli_data, 'sql_injection/sqli_dataset.jsonl')
    
    print(f"\n📊 Dataset Statistics:")
    sqli_positives = sum(1 for x in sqli_data if x['label'] == 1)
    print(f"  SQLi attacks: {sqli_positives}")
    print(f"  Safe payloads: {len(sqli_data) - sqli_positives}")
    print(f"  Total samples: {len(sqli_data)}")
