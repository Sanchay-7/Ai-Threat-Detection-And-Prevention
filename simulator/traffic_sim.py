import asyncio
import random
import argparse
import aiohttp
import time

async def hit(session, base, ip, path="/"):
    """Sends a single HTTP GET request with a spoofed IP header."""
    headers = {"X-Forwarded-For": ip}
    try:
        # Set a reasonable timeout for requests
        async with session.get(base + path, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as r:
            await r.text()
    except Exception:
        pass

async def normal_traffic_generator(base, n_clients=50, rps=20.0):
    """Simulates normal traffic from a pool of random IP addresses."""
    print(f"Starting normal traffic simulation: {n_clients} clients, ~{rps} RPS total.")
    ips = [".".join(str(random.randint(1, 254)) for _ in range(4)) for _ in range(n_clients)]

    async with aiohttp.ClientSession() as session:
        while True:
            ip = random.choice(ips)
            path = random.choice(["/", "/healthz", "/metrics", "/static/app.js"])
            asyncio.create_task(hit(session, base, ip, path=path))
            await asyncio.sleep(1.0 / max(1, rps))

async def ddos_attacker(base, attacker_ip="203.0.113.9", rps=200.0, path="/"):
    """Simulates a high-rate DDoS attack from a single IP address."""
    print(f"Starting DDoS attack from IP {attacker_ip} at {rps} RPS.")
    async with aiohttp.ClientSession() as session:
        while True:
            asyncio.create_task(hit(session, base, attacker_ip, path=path))
            await asyncio.sleep(1.0 / max(1, rps))

async def main():
    parser = argparse.ArgumentParser(description="Traffic simulator for AI DDoS Shield")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="Base URL of the target application")
    parser.add_argument("--normal_ips", type=int, default=50, help="Number of unique IPs for normal traffic")
    parser.add_argument("--normal_rps", type=float, default=20.0, help="Total requests per second for normal traffic")
    parser.add_argument("--attack", action="store_true", help="Enable the DDoS attacker")
    parser.add_argument("--attack_ip", default="203.0.113.9", help="IP address of the attacker")
    parser.add_argument("--attack_rps", type=float, default=200.0, help="Requests per second for the attacker")
    args = parser.parse_args()

    print("--- Traffic Simulator ---")
    tasks = [
        asyncio.create_task(normal_traffic_generator(args.base, args.normal_ips, args.normal_rps))
    ]

    if args.attack:
        print("ATTACK MODE ENABLED.")
        tasks.append(asyncio.create_task(ddos_attacker(args.base, args.attack_ip, args.attack_rps)))

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\nSimulator stopped.")

if __name__ == "__main__":
    asyncio.run(main())
