#!/bin/bash
# PillTrack - Raspberry Pi 5 Setup Script
# ทำให้ง่ายขึ้น!

echo "╔════════════════════════════════════════╗"
echo "║  🔧 PillTrack Raspberry Pi 5 Setup    ║"
echo "╚════════════════════════════════════════╝"
echo ""

# ตรวจสอบว่าเป็น Pi หรือไม่
if [[ ! -f /etc/os-release ]]; then
    echo "❌ ไฟล์ os-release ไม่พบ ตรวจสอบ OS"
    exit 1
fi

# ติดตั้ง system dependencies
echo "📦 ติดตั้ง system dependencies..."
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git
sudo apt install -y libatlas-base-dev libjasper-dev libtiff-dev \
    libharfbuzz0b libwebp6 libharfbuzz-dev libwebp-dev
sudo apt install -y python3-torch python3-torchvision

# สร้าง virtual environment
echo "🔄 สร้าง virtual environment..."
python3 -m venv venv
source venv/bin/activate

# ติดตั้ง Python packages
echo "📚 ติดตั้ง Python packages..."
pip install --upgrade pip
pip install -r requirements_pi5.txt
pip install ultralytics

# ตรวจสอบการติดตั้ง
echo ""
echo "✅ ตรวจสอบการติดตั้ง..."
python3 -c "import torch; print('✅ PyTorch:', torch.__version__)"
python3 -c "import cv2; print('✅ OpenCV:', cv2.__version__)"
python3 -c "import faiss; print('✅ FAISS OK')"
python3 -c "from ultralytics import YOLO; print('✅ YOLO OK')"

echo ""
echo "🎉 การติดตั้งเสร็จ!"
echo ""
echo "⏭️  ขั้นตอนถัดไป:"
echo "1. python3 phase1_database_preparation_pi5.py  (สร้าง database)"
echo "2. python3 phase2_live_inference_pi5.py        (รันจริง)"
echo ""
echo "📖 ดูรายละเอียดที่: README_PI5.md"
