#!/bin/bash
set -e

# 1. Export Google credentials path
export GOOGLE_APPLICATION_CREDENTIALS="/app/gen-lang-client-0305875347-988df3a74aed.json"

# 2. Start the app
echo "[+] Starting Interview Bot..."
python app.py
