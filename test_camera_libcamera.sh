#!/bin/bash

echo "="
echo "🔍 Testing Camera with libcamera-hello"
echo "="

# Test libcamera availability
if ! command -v libcamera-hello &> /dev/null; then
    echo "❌ libcamera-hello not found"
    echo "Install with: sudo apt install -y libcamera-tools"
    exit 1
fi

echo ""
echo "📷 Running libcamera-hello for 3 seconds..."
echo "    (You should see camera output - press Ctrl+C to stop)"
echo ""

timeout 3 libcamera-hello --timeout 3000 2>&1 | head -30

if [ $? -eq 0 ] || [ $? -eq 124 ]; then
    echo ""
    echo "✅ Camera is available and working!"
    echo ""
    exit 0
else
    echo ""
    echo "❌ Camera test failed"
    echo "Try: sudo reboot"
    exit 1
fi
