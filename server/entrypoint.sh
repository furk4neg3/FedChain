#!/bin/sh
set -e

# Guarantee we’re in /app (even if WORKDIR didn’t apply)
cd /app

echo "⏳ waiting for Ganache…"
sleep 5

echo "🚀 Deploying smart contract"
python /app/blockchain/scripts/deploy.py

echo "▶️ Starting Flower Server"
exec python /app/server/server.py
