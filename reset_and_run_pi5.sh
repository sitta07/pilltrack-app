#!/bin/bash
# 🚀 Kill old inference + Release camera + Start new

set -e

echo "╔════════════════════════════════════════════════════╗"
echo "║   🔓 Full Camera Reset & Restart                 ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Step 1: Kill all old Python processes
echo "1️⃣  Killing old processes..."
pkill -f "phase2_live_inference" || true
sleep 1
pkill -f "libcamera" || true
sleep 1

# Step 2: Release camera device
echo "2️⃣  Releasing camera..."
python3 release_camera_pi5.py || true
sleep 2

# Step 3: Check camera is available
echo ""
echo "3️⃣  Checking camera..."
if python3 test_camera_pi5.py; then
    echo ""
    echo "✅ Camera is ready!"
    echo ""
    echo "4️⃣  Starting inference..."
    echo ""
    python3 phase2_live_inference_pi5.py
else
    echo ""
    echo "❌ Camera still not available"
    echo "   Try rebooting: sudo reboot"
    exit 1
fi
