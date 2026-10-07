#!/usr/bin/env bash
# RouteMem AI Gateway — Systemd Service Setup
# Automatically creates and enables routemem-gateway.service for process supervision

set -e

SERVICE_FILE="/etc/systemd/system/routemem-gateway.service"
APP_DIR="/home/ubuntu/routemem-gateway"
VENV_PYTHON="$APP_DIR/.venv/bin/uvicorn"

echo "=== Setting up RouteMem Gateway Systemd Service ==="

cat <<EOF | sudo tee $SERVICE_FILE
[Unit]
Description=RouteMem AI Gateway Service
After=network.target docker.service ollama.service
Wants=docker.service ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$APP_DIR
Environment=PATH=$APP_DIR/.venv/bin:/usr/local/bin:/usr/bin
ExecStart=$APP_DIR/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=3
StandardOutput=append:/var/log/routemem-gateway.log
StandardError=append:/var/log/routemem-gateway-error.log

[Install]
WantedBy=multi-user.target
EOF

echo "[+] Created service file at $SERVICE_FILE"
sudo systemctl daemon-reload
sudo systemctl enable routemem-gateway.service
sudo systemctl restart routemem-gateway.service
echo "[+] RouteMem Gateway service enabled and started!"
sudo systemctl status routemem-gateway.service --no-pager
