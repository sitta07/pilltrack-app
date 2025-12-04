#!/bin/bash
# QUICK COPY-PASTE COMMANDS FOR RASPBERRY PI 5

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   PillTrack Raspberry Pi 5 - Quick Start Commands                    ║
║   Copy & paste these commands into your Pi terminal                  ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝


OPTION 1: FULLY AUTOMATIC (RECOMMENDED)
════════════════════════════════════════════════════════════════════════

# Copy & paste all these lines:

cd ~/Desktop/pilltrack-app
bash install_pi5_python313.sh

# That's it! The script will:
# ✅ Update system packages
# ✅ Install build dependencies
# ✅ Create virtual environment
# ✅ Install NumPy 2.1.3 (fixed)
# ✅ Install PyTorch from apt (fast)
# ✅ Install all other packages
# ✅ Verify installation


OPTION 2: MANUAL (WITH APT)
════════════════════════════════════════════════════════════════════════

# Copy & paste these lines one by one:

# Step 1: Update system
sudo apt update -y
sudo apt upgrade -y

# Step 2: Install build tools
sudo apt install -y python3-dev build-essential python3-venv git

# Step 3: Install NumPy and PyTorch from apt (FAST)
sudo apt install -y python3-numpy python3-torch python3-torchvision

# Step 4: Go to project
cd ~/Desktop/pilltrack-app

# Step 5: Create virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Step 6: Upgrade pip
pip install --upgrade pip setuptools wheel

# Step 7: Install remaining packages
pip install -r requirements_pi5.txt --prefer-binary --no-cache-dir

# Step 8: Install YOLO
pip install ultralytics

# Step 9: Verify
python3 check_pi5_setup.py


OPTION 3: QUICK APT ONLY (FASTEST - 5 MIN)
════════════════════════════════════════════════════════════════════════

# For pure apt installation:

# Install all from system
sudo apt install -y python3-numpy python3-opencv python3-torch python3-torchvision

cd ~/Desktop/pilltrack-app
python3 -m venv myenv
source myenv/bin/activate

pip install -r requirements_pi5.txt --prefer-binary

# Done in ~5 minutes!


VERIFICATION
════════════════════════════════════════════════════════════════════════

# After installation, verify everything works:

python3 check_pi5_setup.py

# Should show:
# ✅ All Pi 5 files verified
# ✅ All documentation present
# ✅ All models found
# ✅ All data folders ready


NEXT STEPS (AFTER INSTALLATION)
════════════════════════════════════════════════════════════════════════

# 1. Create database (run once, takes 2-4 hours)
python3 phase1_database_preparation_pi5.py

# 2. Run live inference (run every time)
python3 phase2_live_inference_pi5.py

# That's it! Drug identification system is ready!


TROUBLESHOOTING
════════════════════════════════════════════════════════════════════════

# If you see numpy build error:
pip install numpy==2.1.3 --prefer-binary --no-cache-dir

# If PyTorch not installed:
sudo apt install -y python3-torch python3-torchvision

# If Camera not found:
libcamera-hello -t 5

# For full troubleshooting guide:
cat NUMPY_BUILD_ERROR_FIX.md


SYSTEM REQUIREMENTS
════════════════════════════════════════════════════════════════════════

# Check you have:
python3 --version           # Should be 3.10+
uname -m                    # Should be aarch64 (ARM 64-bit)
free -h                     # At least 2GB free RAM
df -h                       # At least 5GB free disk


TIME ESTIMATES
════════════════════════════════════════════════════════════════════════

Option 1 (Auto Script):   10-15 minutes
Option 2 (Manual):        10-20 minutes
Option 3 (APT Only):      5-10 minutes (FASTEST)

Database Creation:        2-4 hours (do once)
Live Inference:           Real-time (12-15 FPS)


SAVE THIS FILE
════════════════════════════════════════════════════════════════════════

cp QUICK_START.sh ~/QUICK_START.sh
cat ~/QUICK_START.sh  # to view commands


═══════════════════════════════════════════════════════════════════════════
✅ PillTrack is ready to install on Raspberry Pi 5!
═══════════════════════════════════════════════════════════════════════════

EOF
