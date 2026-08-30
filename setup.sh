#!/usr/bin/env bash
set -e
echo "========================================================="
echo "Space Radiation Hazard Forecast: Environment Setup (POSIX)"
echo "========================================================="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
chmod +x run.sh
echo "[✓] Environment configured. Launch with ./run.sh"
