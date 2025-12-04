#!/bin/bash

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  🚀 ULTIMATE Pi5 CAMERA RECOVERY              ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function log_step() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📍 $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

function log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

function log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

function log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Kill all processes
log_step "STEP 1: Terminating all camera processes"

processes_to_kill=(
    "phase2_live_inference_pi5.py"
    "phase1_database_preparation_pi5.py"
    "libcamera-hello"
    "libcamera-jpeg"
    "libcamera-still"
    "test_camera_pi5.py"
    "check_models_pi5.py"
)

for proc in "${processes_to_kill[@]}"; do
    if pgrep -f "$proc" > /dev/null; then
        log_warning "Killing: $proc"
        pkill -9 -f "$proc" || true
    fi
done

sleep 1

# Kill any remaining Python processes (if user wants aggressive kill)
if [ "$1" == "--aggressive" ]; then
    log_warning "AGGRESSIVE MODE: Killing ALL Python processes"
    pkill -9 python3 || true
    sleep 1
fi

log_success "All processes terminated"

# Step 2: Release device using multiple methods
log_step "STEP 2: Releasing camera device"

# Method 1: Using fuser
if command -v fuser &> /dev/null; then
    log_warning "Using fuser to release /dev/video0"
    sudo fuser -k /dev/video0 2>/dev/null || true
    sleep 1
fi

# Method 2: Using Python script
if [ -f "release_camera_pi5.py" ]; then
    log_warning "Running release_camera_pi5.py"
    python3 release_camera_pi5.py 2>/dev/null || true
    sleep 1
fi

log_success "Camera device released"

# Step 3: Kernel module reset (if on Pi)
log_step "STEP 3: Kernel module reset"

if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null || grep -q "BCM" /proc/device-tree/model 2>/dev/null; then
    log_warning "Pi detected - attempting kernel module reset"
    
    # Unload camera module
    if lsmod | grep -q imx708; then
        log_warning "Unloading imx708 kernel module"
        sudo modprobe -r imx708 2>/dev/null || true
        sleep 2
    fi
    
    if lsmod | grep -q bcm2835_isp; then
        log_warning "Unloading bcm2835_isp kernel module"
        sudo modprobe -r bcm2835_isp 2>/dev/null || true
        sleep 2
    fi
    
    # Reload modules
    log_warning "Reloading camera kernel modules"
    sudo modprobe imx708 2>/dev/null || true
    sleep 1
    
    log_success "Kernel modules reloaded"
else
    log_warning "Not a Pi - skipping kernel module reset"
fi

# Step 4: Wait for device
log_step "STEP 4: Waiting for camera device"

MAX_WAIT=30
WAITED=0

while [ ! -e /dev/video0 ] && [ $WAITED -lt $MAX_WAIT ]; do
    log_warning "Waiting... ($WAITED/$MAX_WAIT seconds)"
    sleep 1
    WAITED=$((WAITED + 1))
done

if [ -e /dev/video0 ]; then
    log_success "Camera device found: /dev/video0"
    ls -la /dev/video0
else
    log_error "Camera device not found after $MAX_WAIT seconds"
    echo ""
    echo "Try: sudo reboot"
    exit 1
fi

# Step 5: Test camera
log_step "STEP 5: Testing camera"

if [ -f "test_camera_minimal.py" ]; then
    if python3 test_camera_minimal.py; then
        log_success "Camera test passed!"
    else
        log_error "Camera test failed"
        echo ""
        echo "Try: sudo reboot"
        exit 1
    fi
else
    log_warning "test_camera_minimal.py not found, skipping test"
fi

# Step 6: Start inference
log_step "STEP 6: Starting inference"

if [ ! -f "phase2_live_inference_pi5.py" ]; then
    log_error "phase2_live_inference_pi5.py not found!"
    exit 1
fi

echo ""
log_success "All systems ready!"
echo ""
echo "Starting: python3 phase2_live_inference_pi5.py"
echo ""

# Run inference with auto-restart
MAX_RESTARTS=3
RESTART_COUNT=0

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 INFERENCE RUN #$((RESTART_COUNT + 1))"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    python3 phase2_live_inference_pi5.py
    INFERENCE_EXIT=$?
    
    if [ $INFERENCE_EXIT -eq 0 ]; then
        log_success "Inference completed successfully"
        exit 0
    else
        RESTART_COUNT=$((RESTART_COUNT + 1))
        
        if [ $RESTART_COUNT -lt $MAX_RESTARTS ]; then
            log_warning "Inference crashed (exit code: $INFERENCE_EXIT)"
            log_warning "Restarting in 3 seconds... (attempt $RESTART_COUNT/$MAX_RESTARTS)"
            sleep 3
            
            # Release camera again before retry
            python3 release_camera_pi5.py 2>/dev/null || true
            sleep 2
        fi
    fi
done

if [ $RESTART_COUNT -eq $MAX_RESTARTS ]; then
    log_error "Inference failed after $MAX_RESTARTS attempts"
    echo ""
    echo "Try: sudo reboot"
    exit 1
fi
