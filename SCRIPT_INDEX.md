# 📑 Pi5 Camera Recovery - Files Index

## 🎬 CAMERA RECOVERY SCRIPTS (USE THESE!)

### ⭐ Main Recovery Script
- **`recover_camera_ultimate.sh`** - RECOMMENDED
  - Auto-kills processes, releases device, resets kernel, tests, starts inference
  - Supports `--aggressive` mode
  - Best option for getting camera working quickly

### 🔓 Process Release Scripts
- **`release_camera_pi5.py`** - Kill stuck processes using lsof
- **`reset_and_run_pi5.sh`** - Step-by-step reset (older version)
- **`reset_and_run_pi5_new.sh`** - Improved step-by-step reset

### 🧪 Testing Scripts
- **`test_camera_minimal.py`** - Lightweight camera test (30 seconds)
- **`test_camera_libcamera.sh`** - libcamera-specific test
- **`test_camera_pi5.py`** - Comprehensive camera test
- **`diagnose_camera.py`** - Full system diagnostic (outputs lots of info)

### 🚀 Quick Utilities
- **`run_inference_pi5.sh`** - Run with auto-restart
- **`check_models_pi5.py`** - Verify model files exist

---

## 📖 DOCUMENTATION (READ THESE!)

### 🎯 START HERE
- **`README_CAMERA.md`** ⭐ - 30-second quick start guide
- **`CAMERA_RECOVERY_GUIDE.md`** - Comprehensive recovery guide with all options

### 🔧 Troubleshooting
- **`CAMERA_TROUBLESHOOTING.md`** - Common errors and solutions
- **`CAMERA_BUSY_FIX.md`** - "Device busy" error solutions

### 📋 Setup & Performance
- **`FIXES_FOR_PI5_DEPLOYMENT.md`** - Pre-deployment checklist
- **`PI5_PERFORMANCE_OPTIMIZATION.md`** - Performance tuning guide
- **`NUMPY_ERROR_FIXED.md`** - Installation fixes

---

## 🐍 MAIN INFERENCE SCRIPTS

### Live Inference (PRODUCTION)
- **`phase2_live_inference_pi5.py`** (676 lines)
  - Optimized for Pi5 with SKIP_DISPLAY=True
  - 320x240 resolution, 12-15 FPS
  - Includes camera retry logic (3 attempts)
  - YOLO detection + EfficientNet feature extraction + FAISS search

### Database Preparation
- **`phase1_database_preparation_pi5.py`** (450 lines)
  - Converts drug images to FAISS searchable database
  - Array stride fixes applied
  - Image dimension validation

---

## ⚙️ CONFIGURATION & INSTALLATION

### Package Management
- **`requirements_pi5.txt`** - All dependencies with working versions
  - numpy==2.1.3, opencv-python==4.10.0.84, etc.
  - Pre-built wheel versions only

### Installation Scripts
- **`install_pi5_python313.sh`** - Automated Pi5 setup
  - Updates build dependencies
  - Installs Python 3.13 packages
  - Sets up FAISS and models

---

## 📚 DECISION TREE

```
Does camera show error?
├─ YES → Run: bash recover_camera_ultimate.sh
└─ NO → Run: python3 phase2_live_inference_pi5.py

Still not working?
├─ Quick test → python3 test_camera_minimal.py
├─ Deep debug → python3 diagnose_camera.py
├─ Manual fix → python3 release_camera_pi5.py
└─ Last resort → sudo reboot
```

---

## 🎯 QUICK COMMAND REFERENCE

| Need | Command |
|------|---------|
| Fix camera error | `bash recover_camera_ultimate.sh` |
| Quick camera test | `python3 test_camera_minimal.py` |
| Full system check | `python3 diagnose_camera.py` |
| Kill stuck processes | `python3 release_camera_pi5.py` |
| Start inference | `python3 phase2_live_inference_pi5.py` |
| Full reboot | `sudo reboot` |
| Check running processes | `ps aux \| grep python` |
| See camera users | `lsof /dev/video0` |

---

## ✅ VERIFICATION CHECKLIST

Before running inference, verify:

```bash
# 1. Camera device exists
ls /dev/video0

# 2. No stuck processes
ps aux | grep python | grep -v grep

# 3. Camera works
python3 test_camera_minimal.py

# 4. Models exist
ls -la phase1_database_preparation_pi5.py
ls -la phase2_live_inference_pi5.py
ls -la faiss_database/

# 5. Dependencies installed
python3 -c "import cv2, torch, faiss; print('OK')"
```

---

## 📊 FILE ORGANIZATION

```
/Users/sittasahathum/Desktop/pilltrack-app/
│
├── 🎬 CAMERA RECOVERY SCRIPTS
│   ├── recover_camera_ultimate.sh ⭐
│   ├── release_camera_pi5.py
│   ├── test_camera_minimal.py
│   ├── diagnose_camera.py
│   └── ... (more test scripts)
│
├── 📖 DOCUMENTATION
│   ├── README_CAMERA.md ⭐
│   ├── CAMERA_RECOVERY_GUIDE.md
│   ├── CAMERA_TROUBLESHOOTING.md
│   └── ... (more guides)
│
├── 🐍 MAIN SCRIPTS
│   ├── phase2_live_inference_pi5.py (main)
│   ├── phase1_database_preparation_pi5.py
│   └── ...
│
├── ⚙️ CONFIGURATION
│   ├── requirements_pi5.txt
│   ├── install_pi5_python313.sh
│   └── ...
│
└── 📦 MODELS & DATA
    ├── best_process_2.onnx
    ├── seg_db_best.pt
    ├── faiss_database/
    └── drug-scraping-c/
```

---

## 🚀 RECOMMENDED WORKFLOW

### First Time Setup
```bash
1. bash recover_camera_ultimate.sh
2. Let it run inference
3. Press Ctrl+C to stop
```

### Regular Usage
```bash
1. python3 phase2_live_inference_pi5.py
   (or bash recover_camera_ultimate.sh if error)
```

### Debugging
```bash
1. python3 diagnose_camera.py
2. Read output for issues
3. Apply fixes from documentation
```

### Deep Issues
```bash
1. python3 release_camera_pi5.py
2. sudo reboot
3. Try again
```

---

## 💡 KEY POINTS

✅ **Do this**:
- Run `recover_camera_ultimate.sh` when camera fails
- Use `test_camera_minimal.py` to verify quickly
- Check `ps aux | grep python` for stuck processes
- Read error messages carefully (they suggest solutions)

❌ **Don't do this**:
- Don't run multiple inference instances
- Don't skip camera test before troubleshooting
- Don't assume camera is broken (usually it's just busy)
- Don't ignore process cleanup

---

## 📞 HELP RESOURCES

| Issue | File to Read |
|-------|--------------|
| Camera won't open | `CAMERA_RECOVERY_GUIDE.md` |
| "Device busy" error | `CAMERA_TROUBLESHOOTING.md` |
| Performance too slow | `PI5_PERFORMANCE_OPTIMIZATION.md` |
| Models not loading | `check_models_pi5.py` |
| Installation errors | `NUMPY_ERROR_FIXED.md` |
| Complete system check | `diagnose_camera.py` |

---

## 🎬 NEXT STEPS

1. **Read**: `README_CAMERA.md` (2 min)
2. **Run**: `bash recover_camera_ultimate.sh` (2 min)
3. **Verify**: See inference output with FPS counter
4. **Success**: 🎉

---

**Questions? All scripts have detailed comments and help output.**
