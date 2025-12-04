#!/bin/bash
# 🚀 Auto-restart inference with camera recovery

set -e

echo "╔════════════════════════════════════════════════════╗"
echo "║   PillTrack Pi 5 - Inference with Auto-Restart    ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

MAX_RETRIES=3
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_RETRIES ]; do
    ATTEMPT=$((ATTEMPT + 1))
    
    echo "📊 Attempt $ATTEMPT/$MAX_RETRIES"
    echo "=================================================="
    echo ""
    
    if python3 phase2_live_inference_pi5.py; then
        echo ""
        echo "✅ Inference completed successfully"
        exit 0
    else
        EXIT_CODE=$?
        echo ""
        echo "⚠️  Inference failed with exit code $EXIT_CODE"
        
        if [ $ATTEMPT -lt $MAX_RETRIES ]; then
            echo ""
            echo "🔓 Releasing camera resources..."
            python3 release_camera_pi5.py
            
            echo ""
            echo "⏳ Waiting before retry..."
            sleep 3
        else
            echo ""
            echo "❌ All retries failed"
            echo ""
            echo "📋 Try these steps:"
            echo "   1. Release camera: python3 release_camera_pi5.py"
            echo "   2. Check camera: python3 test_camera_pi5.py"
            echo "   3. Reboot: sudo reboot"
            exit 1
        fi
    fi
done
