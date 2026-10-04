#!/bin/bash
# RouteMem AI Gateway — AWS Control Plane Bootstrap Script (t4g.xlarge / Ubuntu 22.04 ARM64)
set -e

echo "=== Bootstrapping RouteMem Control Plane Node ==="

# 1. System Updates & Prerequisites
sudo apt-get update
sudo apt-get install -y docker.io docker-compose python3.11 python3-venv git curl awscli

# 2. Add current user to Docker group
sudo usermod -aG docker ubuntu || true

# 3. Create Project Directory
mkdir -p ~/routemem-gateway
cd ~/routemem-gateway

# 4. Set up Virtual Environment
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# 5. Install Dependencies
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "=== Control Plane Setup Complete! ==="
