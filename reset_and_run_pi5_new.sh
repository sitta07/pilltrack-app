#!/bin/bash

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  🚀 Pi5 CAMERA RESET & INFERENCE     ║"
echo "╚═══════════════════════════════════════╝"
echo ""

# Check if running on Pi
if ! command -v vcgencmd &> /dev/null; then
    echo "⚠️  Warning: Not running on Raspberry Pi"
fi

# Step 1: Kill stuck processes
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Killing stuck processes..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 release_camera_pi5.py
if [ $? -ne 0 ]; then
    echo "⚠️  Release script failed, continuing..."
fi

sleep 2

# Step 2: Test camera with libcamera
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Testing camera with libcamera-hello..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v libcamera-hello &> /dev/null; then
    echo ""
    timeout 3 libcamera-hello --timeout 2000 2>&1 | head -20
    
    if [ $? -eq 0 ] || [ $? -eq 124 ]; then
        echo "✅ Camera is working!"
    else
        echo "⚠️  libcamera test inconclusive, trying Python..."
    fi
else
    echo "⚠️  libcamera-hello not found, skipping libcamera test"
fi

sleep 1

# Step 3: Test camera with Python
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Testing camera with Python..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 -c "
import cv2
print('Opening camera...')
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f'✅ Camera working! Frame size: {frame.shape}')
    else:
        print('⚠️  Could not read frame')
    cap.release()
else:
    print('❌ Could not open camera')
    exit(1)
"

CAMERA_TEST=$?

if [ $CAMERA_TEST -ne 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ CAMERA NOT WORKING"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Try these solutions:"
    echo ""
    echo "1. Full kernel reset:"
    echo "   sudo reboot"
    echo ""
    echo "2. Manual process kill:"
    echo "   pkill -9 python3"
    echo "   pkill -9 libcamera"
    echo "   fuser -k /dev/video0"
    echo ""
    echo "3. Check camera permissions:"
    echo "   ls -la /dev/video0"
    echo ""
    exit 1
fi

sleep 1

# Step 4: Run inference
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Starting inference..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 phase2_live_inference_pi5.py

# If inference stops, run it again (auto-restart)
INFERENCE_EXIT=$?
if [ $INFERENCE_EXIT -ne 0 ]; then
    echo ""
    echo "⚠️  Inference stopped (exit code: $INFERENCE_EXIT)"
    echo ""
    echo "Retrying in 2 seconds..."
    sleep 2
    
    echo ""
    python3 release_camera_pi5.py
    sleep 2
    
    echo ""
    python3 phase2_live_inference_pi5.py
fi
