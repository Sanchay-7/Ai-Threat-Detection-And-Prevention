# AI-Powered DDoS Hybrid Shield

This project is a FastAPI application designed to detect and prevent DDoS attacks using a hybrid approach that combines machine learning, signature analysis, rate limiting, and a real firewall (`iptables`) for blocking malicious IP addresses.

## Features

-   **Hybrid Detection:** Uses a combination of a Random Forest classifier, an Isolation Forest for anomaly detection, a neural MLP classifier with feature scaling, and substring-based signature matching.
-   **Real-time Firewalling:** Directly integrates with Linux `iptables` to block and unblock IP addresses at the kernel level.
-   **Rate Limiting:** A token-bucket algorithm limits the number of requests per IP.
-   **Live Dashboard:** A real-time frontend built with vanilla JS and Chart.js that visualizes traffic, blocked IPs, and security events via WebSockets.
-   **Relevant ML Model:** Includes a script to generate a synthetic dataset based on realistic HTTP traffic patterns, ensuring the models are trained on features the application can actually observe.

## How It Works

1.  **Middleware:** Every incoming request is intercepted by a FastAPI middleware.
2.  **Filtering:** The middleware first checks if the source IP is already blocked by `iptables` or is exceeding the rate limit.
3.  **Feature Extraction:** For allowed requests, it calculates features in real-time (e.g., request rate from the IP, number of unique paths visited).
4.  **Hybrid Analysis:** These features are fed into the hybrid detector:
    -   The ML models (Supervised and Anomaly) calculate a threat score.
    -   The request payload is scanned for malicious signatures.
5.  **Blocking:** If the detector flags the request as malicious, the source IP is blocked using `iptables` for a configured duration, and a security event is logged.
6.  **Live Updates:** A background task pushes metrics and event logs to all connected dashboard clients via WebSockets.

## Setup and Installation

### 1. Prerequisites

-   A Linux environment (tested on Kali Linux/Debian).
-   Python 3.8+ and `pip`.
-   Root or `sudo` access.

### 2. Sudo Configuration (CRITICAL)

The application needs permission to run `iptables` without a password prompt.

1.  Open the sudoers file: `sudo visudo`
2.  Add this line at the end, replacing `your_user` with your username:
    ```
    your_user ALL=(ALL) NOPASSWD: /usr/sbin/iptables
    ```
3.  Save the file. This allows your user to execute `iptables` with `sudo` without being asked for a password.

### 3. Application Setup

1.  **Clone the repository or create the files from the provided code.**

2.  **Make the run script executable:**
    ```bash
    chmod +x run.sh
    ```

3.  **Generate the Dataset:**
    The ML models need to be trained on data that reflects the features we can extract from HTTP traffic.
    ```bash
    python3 generate_dataset.py
    ```
    This will create `dataset/generated_traffic.csv`.

4.  **Train the Models:**
    ```bash
    python3 train.py
    ```
    This will train the models and save `ddos_supervised.pkl`, `ddos_anom.pkl`, and `ddos_mlp.pkl`.

### 4. Running the Application

**Important:** You must use `sudo` to allow the application to interact with `iptables`.

```bash
sudo ./run.sh
