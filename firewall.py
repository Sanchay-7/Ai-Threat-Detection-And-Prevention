import subprocess
import logging
import time
import os
from typing import Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FirewallManager:
    def __init__(self):
        # Use a set for fast lookups of currently blocked IPs
        self.blocked_ips: Set[str] = set()
        self.skip = os.environ.get("SKIP_IPTABLES", "0") == "1"
        if self.skip:
            logging.info("SKIP_IPTABLES=1 set; firewall sync disabled (development mode).")
        else:
            self._sync_from_iptables()

    def _run_command(self, command: list[str]) -> bool:
        """Runs a shell command, logs output, and returns success status."""
        if self.skip:
            logging.info(f"[SKIP_IPTABLES] Would run: {' '.join(command)}")
            return True
        try:
            # We must use 'sudo' to modify iptables
            full_command = ['sudo'] + command
            process = subprocess.run(
                full_command,
                check=True,
                capture_output=True,
                text=True
            )
            logging.info(f"Successfully executed: {' '.join(full_command)}")
            if process.stdout:
                logging.debug(f"stdout: {process.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing: {' '.join(e.cmd)}")
            logging.error(f"Return code: {e.returncode}")
            logging.error(f"Stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            logging.error("Error: 'sudo' or 'iptables' command not found. Is it in the system's PATH?")
            return False

    def _sync_from_iptables(self):
        """Syncs the internal state with the actual iptables rules."""
        logging.info("Syncing blocked IPs from iptables rules...")
        self.blocked_ips.clear()
        if self.skip:
            logging.info("SKIP_IPTABLES=1; skipping iptables sync.")
            return
        try:
            # List all rules in the INPUT chain
            result = subprocess.run(
                ['sudo', 'iptables', '-L', 'INPUT', '-n'],
                capture_output=True, text=True, check=True
            )
            for line in result.stdout.splitlines():
                # Look for lines that indicate a DROP rule for a specific source IP
                if 'DROP' in line and '0.0.0.0/0' not in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        ip = parts[3]
                        self.blocked_ips.add(ip)
            logging.info(f"Sync complete. Found {len(self.blocked_ips)} blocked IPs.")
        except Exception as e:
            logging.error(f"Could not sync from iptables: {e}")

    def block_ip(self, ip: str) -> bool:
        """Blocks a given IP address using iptables."""
        if ip in self.blocked_ips:
            logging.warning(f"IP {ip} is already blocked.")
            return True
        
        # We insert the rule at the top of the INPUT chain
        command = ['iptables', '-I', 'INPUT', '1', '-s', ip, '-j', 'DROP']
        success = self._run_command(command)
        if success:
            self.blocked_ips.add(ip)
            logging.info(f"Successfully blocked IP: {ip}")
        return success

    def unblock_ip(self, ip: str) -> bool:
        """Unblocks a given IP address by deleting the corresponding iptables rule."""
        if ip not in self.blocked_ips:
            logging.warning(f"IP {ip} is not in the active block list.")
            return True
            
        command = ['iptables', '-D', 'INPUT', '-s', ip, '-j', 'DROP']
        success = self._run_command(command)
        if success:
            if ip in self.blocked_ips:
                self.blocked_ips.remove(ip)
            logging.info(f"Successfully unblocked IP: {ip}")
        return success

    def is_blocked(self, ip: str) -> bool:
        """Checks if an IP is in the set of blocked IPs."""
        return ip in self.blocked_ips

# Create a global instance to be used by the app
firewall = FirewallManager()
