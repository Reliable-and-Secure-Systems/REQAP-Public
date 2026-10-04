#!/usr/bin/env bash
# ==============================================================================
# run_demo.sh - 1-Click Demonstration Launcher for Linux (Ubuntu / Debian)
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo " ⚡ Xilinx VC707 FPGA Edge-AI Live Demonstration"
echo "========================================================"
echo ""

# Check python3
if ! command -v python3 &> /dev/null; then
    echo "[-] Error: python3 is not installed or not in PATH."
    exit 1
fi

echo "[+] Checking & Installing Python dependencies..."
python3 -m pip install -q -r requirements.txt

echo "[+] Launching Streamlit Demonstration Dashboard..."
echo "[+] Opening browser at http://localhost:8501"
echo ""

python3 -m streamlit run app.py
