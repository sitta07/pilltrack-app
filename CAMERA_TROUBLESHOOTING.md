# 🎥 Pi5 Camera Troubleshooting Guide

## 🆘 Common Errors & Solutions

### Error 1: "Device or resource busy"

**Symptom:**
```
RuntimeError: Failed to acquire camera: Device or resource busy
```

**Solution A: Release stuck processes (RECOMMENDED)**
```bash
python3 release_camera_pi5.py
python3 phase2_live_inference_pi5.py
```

**Solution B: Full diagnostic + reset**
```bash
python3 diagnose_camera.py
bash reset_and_run_pi5_new.sh
```

**Solution C: Manual kernel reset**
```bash
# Kill all Python processes
pkill -9 python3

# Release device
sudo fuser -k /dev/video0

# Unload/reload camera module
sudo modprobe -r imx708
sleep 1
sudo modprobe imx708

# Wait 3 seconds
sleep 3

# Try again
python3 phase2_live_inference_pi5.py
```

**Solution D: Full reboot (NUCLEAR OPTION)**
```bash
sudo reboot
```

---

### Error 2: "Cannot read from camera"

**Symptom:**
```
OpenCV Error: Cannot read from camera
CAP_PROP_FPS: 0
```

**Causes:**
- Camera not connected or damaged
- Ribbon cable loose
- Camera module faulty
- Wrong device (/dev/video1 instead of /dev/video0)

**Check Device:**
```bash
ls -la /dev/video*
```

**Check Connection:**
```bash
libcamera-hello --list-properties 2>&1 | head -20
```

**Reset Camera Ribbon:**
- Turn off Pi5
- Reseat the camera ribbon cable (CSI port)
- Turn on Pi5

---

### Error 3: "No module named 'picamera2'"

**Solution:**
```bash
sudo apt update
sudo apt install -y python3-picamera2
sudo apt install -y libcamera-tools
```

---

### Error 4: "No module named 'cv2'"

**Solution:**
```bash
pip3 install --prefer-binary opencv-python==4.10.0.84
```

---

## 🔍 Diagnostic Commands

### Check Camera Device Status
```bash
ls -la /dev/video0
```

### List Processes Using Camera
```bash
lsof /dev/video0
fuser /dev/video0
```

### Test Camera with libcamera
```bash
libcamera-hello --timeout 3000
```

### Test Camera with OpenCV
```bash
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    print(f'Frame: {frame.shape}')
    cap.release()
else:
    print('Camera failed')
"
```

### Full System Diagnostic
```bash
python3 diagnose_camera.py
```

---

## 🚀 Quick Recovery Script

Save as `quick_fix.sh`:
```bash
#!/bin/bash
echo "🔧 Quick Camera Fix..."
python3 release_camera_pi5.py
sleep 2
python3 diagnose_camera.py
sleep 2
python3 phase2_live_inference_pi5.py
```

Then:
```bash
chmod +x quick_fix.sh
./quick_fix.sh
```

---

## 📋 Pre-Check Checklist

Before running inference:

- [ ] Camera module physically connected
- [ ] Ribbon cable properly seated in CSI port
- [ ] No red lights on camera module
- [ ] `ls -la /dev/video0` shows device exists
- [ ] No old Python processes: `ps aux | grep python`
- [ ] `python3 diagnose_camera.py` shows ✅ for camera test
- [ ] At least 100MB free disk space
- [ ] Python 3.11+ installed

---

## 🎯 Which Script to Use?

| Situation | Command |
|-----------|---------|
| First time setup | `python3 diagnose_camera.py` |
| Camera error | `python3 release_camera_pi5.py` |
| Quick restart | `bash reset_and_run_pi5_new.sh` |
| Deep debugging | `python3 diagnose_camera.py` (full output) |
| Manual recovery | `pkill -9 python3` + `sudo reboot` |

---

## 🔧 Advanced Debugging

### Check Kernel Logs
```bash
dmesg | tail -50
```

### Check Camera Kernel Module
```bash
lsmod | grep imx708
modinfo imx708
```

### Check libcamera Config
```bash
cat /etc/libcamera/pisp.json
```

### Monitor Camera in Real-Time
```bash
watch 'lsof /dev/video0'
```

---

## 📞 Still Not Working?

1. Try **Solution C** (manual kernel reset)
2. If still fails, **reboot**: `sudo reboot`
3. If still fails after reboot:
   - Check camera ribbon is firmly seated
   - Try different USB port if using USB camera
   - Check `/var/log/syslog` for hardware errors
   - Consider camera module replacement

---

## 📈 Performance After Fix

Expected performance with `SKIP_DISPLAY=True`:
- **FPS**: 12-15 fps (was 2-3 fps with display on)
- **Resolution**: 320x240 (optimized from 640x480)
- **Detection Time**: 50-80ms per frame
- **CPU Usage**: 70-85%

If you see lower FPS:
- Check CPU temperature: `vcgencmd measure_temp`
- Check for other processes: `top`
- Reduce resolution further in phase2_live_inference_pi5.py (line 41)
