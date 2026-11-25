#!/usr/bin/env bash
# One-time server setup for the ECG API on Ubuntu 24.04 (EC2).
# Installs Docker, adds a 2 GB swap file (t3.micro has 1 GB RAM) and lets the
# default user run docker. Safe to run more than once.
#
# Used automatically as EC2 user-data by deploy/aws_create_ec2.ps1, or by hand:
#   sudo bash ec2_setup.sh
set -euo pipefail

APP_USER="${APP_USER:-ubuntu}"

echo "==> Installing Docker"
if ! command -v docker >/dev/null 2>&1; then
  apt-get update -y
  DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io curl
fi
systemctl enable --now docker

if id "$APP_USER" >/dev/null 2>&1; then
  usermod -aG docker "$APP_USER"
fi

echo "==> Swap"
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> Docker log rotation (keeps the small disk from filling up)"
if [ ! -f /etc/docker/daemon.json ]; then
  cat > /etc/docker/daemon.json <<'EOF'
{ "log-driver": "json-file", "log-opts": { "max-size": "10m", "max-file": "3" } }
EOF
  systemctl restart docker
fi

touch /var/log/ecg-setup-done
echo "==> Done: $(docker --version)"
