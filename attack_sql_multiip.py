#!/usr/bin/env python3
"""
Multi-IP SQL Injection Attack Simulator
Simulates SQL injection attacks from different source IPs.
FOR AUTHORIZED TESTING ONLY - Use on your own systems only.
"""
import requests
import time
import sys
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
API_URL = "http://127.0.0.1:8000/test"
PAYLOADS = [
    "admin' OR '1'='1",
    "' OR 1=1 --",
    "admin' --",
    "' UNION SELECT NULL,NULL--",
    "' AND SLEEP(5)--",
    "'; DROP TABLE users;--",
    "' OR 'x'='x",
    "admin' /*",
    "' or 1=1#",
    "' or 'a'='a",
]

class MultiIPAttackSimulator:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.results = {
            "total": 0,
            "detected": 0,
            "blocked": 0,
            "by_ip": {},
            "payloads": []
        }

    def get_ip_addresses(self, count: int = 10) -> List[str]:
        """Generate diverse IP addresses for testing"""
        ips = []
        for i in range(count):
            # Simulate different networks: 192.168.x.x, 10.x.x.x, 172.16.x.x
            if i < count // 3:
                ip = f"192.168.{i}.{100 + i}"
            elif i < 2 * count // 3:
                ip = f"10.{i % 256}.{(i*2) % 256}.{100 + i}"
            else:
                ip = f"172.16.{i % 256}.{100 + i}"
            ips.append(ip)
        return ips

    def test_payload_from_ip(self, payload: str, source_ip: str, request_id: int) -> Dict:
        """Test a payload with spoofed source IP header"""
        try:
            headers = {
                'X-Forwarded-For': source_ip,
                'X-Real-IP': source_ip,
                'CF-Connecting-IP': source_ip
            }
            response = requests.post(
                self.api_url,
                data=payload,
                headers=headers,
                timeout=5
            )
            if response.status_code != 200:
                return {
                    'success': False,
                    'source_ip': source_ip,
                    'payload': payload[:50],
                    'error': f'HTTP {response.status_code}: {response.text[:100]}',
                    'request_id': request_id,
                    'status_code': response.status_code
                }
            result = response.json()
            detected = result.get('sql', {}).get('decision', False)
            
            return {
                'success': True,
                'source_ip': source_ip,
                'payload': payload[:50],
                'detected': detected,
                'score': result.get('sql', {}).get('score', 0),
                'status_code': response.status_code,
                'reason': result.get('sql', {}).get('reason', 'unknown'),
                'request_id': request_id
            }
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'source_ip': source_ip,
                'payload': payload[:50],
                'error': 'Connection refused (IP blocked?)',
                'request_id': request_id,
                'status_code': 0
            }
        except Exception as e:
            return {
                'success': False,
                'source_ip': source_ip,
                'payload': payload[:50],
                'error': str(e),
                'request_id': request_id
            }

    def run_sequential_attack(self, source_ips: List[str], payloads: List[str], 
                             requests_per_ip: int = 5, delay: float = 0.1):
        """Sequential attack from multiple IPs"""
        print(f"\n{'='*80}")
        print(f"🔴 SEQUENTIAL MULTI-IP SQL INJECTION ATTACK")
        print(f"{'='*80}")
        print(f"  Source IPs: {len(source_ips)}")
        print(f"  Payloads per IP: {requests_per_ip}")
        print(f"  Delay between requests: {delay}s")
        print(f"  Total requests: {len(source_ips) * requests_per_ip}\n")
        
        request_id = 0
        for ip in source_ips:
            print(f"🎯 Attacking from IP: {ip}")
            if ip not in self.results["by_ip"]:
                self.results["by_ip"][ip] = {"detected": 0, "blocked": 0, "total": 0}
            
            for i in range(requests_per_ip):
                payload = payloads[i % len(payloads)]
                result = self.test_payload_from_ip(payload, ip, request_id)
                request_id += 1
                
                if result['success']:
                    self.results["detected"] += 1 if result.get('detected') else 0
                    self.results["by_ip"][ip]["detected"] += 1 if result.get('detected') else 0
                    status = "✅ DETECTED" if result.get('detected') else "⚠️  MISSED"
                    print(f"    [{i+1}/{requests_per_ip}] {status} - Score: {result.get('score', 0):.3f}")
                else:
                    status = "🚫 BLOCKED" if "blocked" in result.get('error', '').lower() else "❌ ERROR"
                    self.results["blocked"] += 1 if "blocked" in result.get('error', '').lower() else 0
                    self.results["by_ip"][ip]["blocked"] += 1 if "blocked" in result.get('error', '').lower() else 0
                    print(f"    [{i+1}/{requests_per_ip}] {status} - {result.get('error', 'Unknown error')}")
                
                self.results["by_ip"][ip]["total"] += 1
                self.results["total"] += 1
                self.results["payloads"].append(result)
                
                time.sleep(delay)
            
            print()

    def run_parallel_attack(self, source_ips: List[str], payloads: List[str], 
                           requests_per_ip: int = 5, max_workers: int = 5):
        """Parallel attack from multiple IPs (DDoS-like)"""
        print(f"\n{'='*80}")
        print(f"🔴 PARALLEL MULTI-IP SQL INJECTION ATTACK (DDoS-style)")
        print(f"{'='*80}")
        print(f"  Source IPs: {len(source_ips)}")
        print(f"  Payloads per IP: {requests_per_ip}")
        print(f"  Max concurrent threads: {max_workers}")
        print(f"  Total requests: {len(source_ips) * requests_per_ip}\n")
        
        tasks = []
        request_id = 0
        
        for ip in source_ips:
            if ip not in self.results["by_ip"]:
                self.results["by_ip"][ip] = {"detected": 0, "blocked": 0, "total": 0}
            
            for i in range(requests_per_ip):
                payload = payloads[i % len(payloads)]
                tasks.append((ip, payload, request_id))
                request_id += 1
        
        print(f"Sending {len(tasks)} concurrent requests...\n")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.test_payload_from_ip, payload, ip, rid): (ip, rid)
                for ip, payload, rid in tasks
            }
            
            completed = 0
            for future in as_completed(futures):
                ip, rid = futures[future]
                result = future.result()
                completed += 1
                
                if result['success']:
                    self.results["detected"] += 1 if result.get('detected') else 0
                    self.results["by_ip"][ip]["detected"] += 1 if result.get('detected') else 0
                    status = "✅" if result.get('detected') else "⚠️"
                else:
                    self.results["blocked"] += 1 if "blocked" in result.get('error', '').lower() else 0
                    self.results["by_ip"][ip]["blocked"] += 1 if "blocked" in result.get('error', '').lower() else 0
                    status = "🚫" if "blocked" in result.get('error', '').lower() else "❌"
                
                self.results["by_ip"][ip]["total"] += 1
                self.results["total"] += 1
                self.results["payloads"].append(result)
                
                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{len(tasks)} requests completed")

    def print_summary(self):
        """Print attack summary"""
        print(f"\n{'='*80}")
        print(f"📊 ATTACK SUMMARY")
        print(f"{'='*80}")
        print(f"Total requests sent: {self.results['total']}")
        print(f"Attacks detected: {self.results['detected']}")
        print(f"IPs blocked: {self.results['blocked']}")
        
        if self.results['total'] > 0:
            detection_rate = 100 * self.results['detected'] / self.results['total']
            block_rate = 100 * self.results['blocked'] / self.results['total']
            print(f"\nDetection rate: {detection_rate:.1f}%")
            print(f"Block rate: {block_rate:.1f}%")
        
        print(f"\n📍 Per-IP Statistics:")
        print(f"{'IP':<20} {'Requests':<12} {'Detected':<12} {'Blocked':<12}")
        print(f"{'-'*56}")
        for ip, stats in sorted(self.results["by_ip"].items()):
            print(f"{ip:<20} {stats['total']:<12} {stats['detected']:<12} {stats['blocked']:<12}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-IP SQL Injection Attack Simulator"
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/test",
        help="API endpoint URL"
    )
    parser.add_argument(
        "--ips",
        type=int,
        default=10,
        help="Number of different source IPs (default: 10)"
    )
    parser.add_argument(
        "--requests-per-ip",
        type=int,
        default=5,
        help="Number of requests from each IP (default: 5)"
    )
    parser.add_argument(
        "--mode",
        choices=['sequential', 'parallel'],
        default='sequential',
        help="Attack mode: sequential or parallel (default: sequential)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent threads for parallel mode (default: 5)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between sequential requests in seconds (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    simulator = MultiIPAttackSimulator(args.api_url)
    ips = simulator.get_ip_addresses(args.ips)
    
    try:
        if args.mode == 'sequential':
            simulator.run_sequential_attack(ips, PAYLOADS, args.requests_per_ip, args.delay)
        else:
            simulator.run_parallel_attack(ips, PAYLOADS, args.requests_per_ip, args.workers)
        
        simulator.print_summary()
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Attack interrupted by user")
        simulator.print_summary()
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
