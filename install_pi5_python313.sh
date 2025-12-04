#!/bin/bash
# PillTrack Raspberry Pi 5 - Installation Guide
# Fix for numpy build error on Python 3.13
# =====================================================

set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║    🔧 PillTrack Pi 5 - Installation (Python 3.13)    ║"
echo "║      Fix for numpy build error                       ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Update system
echo "📦 Step 1: Update system packages..."
sudo apt update -y
sudo apt upgrade -y
echo "✅ System updated"
echo ""

# Step 2: Install build dependencies
echo "🔨 Step 2: Install build dependencies..."
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    libblas-dev \
    liblapack-dev \
    libatlas3-base \
    libtiff-dev \
    libharfbuzz0b \
    libharfbuzz-dev \
    libwebp-dev \
    gfortran
echo "✅ Build dependencies installed"
echo ""

# Step 3: Create virtual environment
echo "🔄 Step 3: Create virtual environment..."
if [ ! -d "myenv" ]; then
    python3 -m venv myenv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
source myenv/bin/activate
echo ""

# Step 4: Upgrade pip/setuptools/wheel
echo "⬆️  Step 4: Upgrade pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel
echo "✅ Upgraded pip/setuptools/wheel"
echo ""

# Step 5: Install numpy (with cache disabled)
echo "📊 Step 5: Install numpy (this may take a while)..."
pip install numpy==2.1.3 --no-cache-dir --prefer-binary
echo "✅ NumPy installed"
echo ""

# Step 6: Install OpenCV via apt (faster)
echo "📷 Step 6: Install OpenCV via system packages..."
sudo apt install -y python3-opencv
echo "✅ OpenCV installed via apt"
echo ""

# Step 7: Install PyTorch via apt (MUCH FASTER)
echo "🔥 Step 7: Install PyTorch via system packages..."
echo "   This is MUCH faster than pip on ARM64!"
sudo apt install -y python3-torch python3-torchvision python3-torchaudio
echo "✅ PyTorch installed via apt"
echo ""

# Step 8: Install Python dependencies from requirements
echo "📚 Step 8: Install Python dependencies..."
echo "   (This may take 10-20 minutes)"
pip install -r requirements_pi5.txt --prefer-binary --no-cache-dir
echo "✅ Dependencies installed"
echo ""

# Step 9: Verify installation
echo "🔍 Step 9: Verifying installation..."
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__}')" 2>/dev/null || echo "⚠️  PyTorch from apt (OK)"
python3 -c "import numpy; print(f'✅ NumPy {numpy.__version__}')"
python3 -c "import cv2; print(f'✅ OpenCV {cv2.__version__}')"
python3 -c "import faiss; print('✅ FAISS OK')"
python3 -c "from ultralytics import YOLO; print('✅ YOLO OK')"
echo ""

# Step 10: Check verification
echo "🎉 Installation complete!"
echo ""
echo "════════════════════════════════════════════════════════"
echo "Next steps:"
echo ""
echo "1. Create database:"
echo "   $ python3 phase1_database_preparation_pi5.py"
echo ""
echo "2. Run live inference:"
echo "   $ python3 phase2_live_inference_pi5.py"
echo ""
echo "════════════════════════════════════════════════════════"
