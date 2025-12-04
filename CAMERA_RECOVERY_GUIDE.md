# 🎬 Pi5 Camera Issue - Complete Recovery Guide

> **Status**: Camera device contention issue detected and fixed
> **Generated**: Latest update
> **Target**: Raspberry Pi 5 with PillTrack drug detection system

---

## 🚨 The Problem

When running `phase2_live_inference_pi5.py`, you get:
```
RuntimeError: Failed to acquire camera: Device or resource busy
Error: -15 (MMAL_STATUS_RESOURCE_USED)
```

**Root Cause**: Another process (or stuck instance) is holding the `/dev/video0` device

---

## ✅ Quick Fix (30 seconds)

### Option 1: Auto Recovery (RECOMMENDED)
```bash
bash recover_camera_ultimate.sh
```

This will:
1. Kill all stuck processes
2. Release device with `fuser`
3. Reset kernel modules
4. Test camera
5. Start inference
6. Auto-restart on failure

### Option 2: Fast Minimal Test
```bash
python3 release_camera_pi5.py
python3 test_camera_minimal.py
python3 phase2_live_inference_pi5.py
```

### Option 3: Aggressive Reset
```bash
bash recover_camera_ultimate.sh --aggressive
```

Kills ALL Python processes (more aggressive)

---

## 📋 Recovery Scripts Provided

| Script | Purpose | When to Use |
|--------|---------|------------|
| `recover_camera_ultimate.sh` | 🏆 **Complete auto-recovery** | First try this |
| `release_camera_pi5.py` | Kill stuck processes | Manual control |
| `test_camera_minimal.py` | Quick camera test | Verify camera works |
| `diagnose_camera.py` | Full system diagnostic | Deep debugging |
| `reset_and_run_pi5_new.sh` | Step-by-step reset | Educational/debugging |
| `test_camera_libcamera.sh` | libcamera test | Low-level camera check |

---

## 🔍 Diagnosis Commands

### See what's using the camera:
```bash
lsof /dev/video0        # Show processes
fuser /dev/video0       # Show PIDs
ps aux | grep video     # Find video processes
```

### Check if camera is working:
```bash
python3 test_camera_minimal.py
libcamera-hello --timeout 2000
```

### Full system report:
```bash
python3 diagnose_camera.py
```

---

## 🔧 Manual Recovery Steps

If scripts don't work:

### Step 1: Kill processes
```bash
pkill -9 python3            # Kill all Python
pkill -9 libcamera          # Kill libcamera
pkill -9 raspistill         # Kill raspistill
```

### Step 2: Release device
```bash
sudo fuser -k /dev/video0
```

### Step 3: Reset kernel module
```bash
sudo modprobe -r imx708
sleep 1
sudo modprobe imx708
sleep 1
sudo modprobe bcm2835_isp
sleep 3
```

### Step 4: Test and run
```bash
python3 test_camera_minimal.py
python3 phase2_live_inference_pi5.py
```

### Step 5: If still fails (NUCLEAR)
```bash
sudo reboot
```

---

## 📊 What Each Recovery Script Does

### recover_camera_ultimate.sh (⭐ Recommended)
```
┌─────────────────────────────┐
│ Kill all processes          │
│ Release /dev/video0         │
│ Reset kernel modules        │
│ Wait for device             │
│ Test camera                 │
│ Start inference             │
│ Auto-restart on failure     │
└─────────────────────────────┘
```

### release_camera_pi5.py
```
┌─────────────────────────────┐
│ Find processes with lsof    │
│ Kill by PID                 │
│ Unload/reload modules       │
│ Wait 3 seconds              │
│ Show next steps             │
└─────────────────────────────┘
```

### test_camera_minimal.py
```
┌─────────────────────────────┐
│ Kill processes              │
│ Test with libcamera         │
│ Test with OpenCV            │
│ Show success/failure        │
└─────────────────────────────┘
```

### diagnose_camera.py
```
┌─────────────────────────────┐
│ System info                 │
│ Kernel modules              │
│ Camera devices              │
│ libcamera support           │
│ OpenCV support              │
│ Picamera2 support           │
│ ML frameworks               │
│ FULL camera test            │
└─────────────────────────────┘
```

---

## 🎯 Which Script to Use?

**First time / "just fix it":**
```bash
bash recover_camera_ultimate.sh
```

**Debugging / need details:**
```bash
python3 diagnose_camera.py
```

**Quick test:**
```bash
python3 test_camera_minimal.py
```

**Manual control:**
```bash
python3 release_camera_pi5.py
python3 phase2_live_inference_pi5.py
```

---

## 🚀 Performance After Recovery

When camera works, expect:

| Metric | Value |
|--------|-------|
| FPS | 12-15 |
| Resolution | 320x240 |
| Detection time | 50-80ms |
| CPU usage | 70-85% |
| Display | OFF (SKIP_DISPLAY=True) |

If lower, check:
- CPU temp: `vcgencmd measure_temp`
- Running processes: `top`
- Disk space: `df -h`

---

## ❌ Still Not Working?

### Last resort options:

**Option A: Aggressive reset**
```bash
bash recover_camera_ultimate.sh --aggressive
```

**Option B: Full reboot**
```bash
sudo reboot
```

**Option C: Check hardware**
- Reseat camera ribbon cable (CSI port)
- Check for bent pins
- Try test with: `libcamera-hello --list-properties`

**Option D: Check logs**
```bash
dmesg | tail -50
cat /var/log/syslog | tail -50
```

---

## 📝 Pre-Check Checklist

Before running inference:

- [ ] Camera physically connected
- [ ] No red error lights on module
- [ ] `ls -la /dev/video0` shows device
- [ ] `python3 test_camera_minimal.py` shows ✅
- [ ] No old Python processes: `ps aux | grep python`
- [ ] Free disk space: `df -h` (> 100MB)
- [ ] Network stable (for Hugging Face downloads)

---

## 🎓 Understanding the Problem

### Why "Device or resource busy"?

1. **Process still holding device**
   - Old inference process didn't cleanup properly
   - Stuck libcamera instance

2. **Device not released after crash**
   - Inference crashed while camera was open
   - Python didn't call `cap.release()`

3. **Multiple instances trying to use camera**
   - Two inference scripts running
   - libcamera and OpenCV both trying to use camera

4. **Kernel module issue**
   - Camera module needs reload
   - Device tree issue

### Why the scripts work:

1. **Kill processes**: Force cleanup of any stuck process
2. **Release device**: Use `fuser` to forcefully release
3. **Reset modules**: Reload kernel drivers from scratch
4. **Test before run**: Verify device is actually available
5. **Auto-restart**: Recover from transient failures

---

## 💡 Prevention Tips

### Avoid this error in the future:

1. **Always clean up camera**
   ```python
   try:
       # use camera
   finally:
       cap.release()  # ALWAYS call this
   ```

2. **Kill old processes before running**
   ```bash
   pkill -f phase2_live_inference || true
   python3 phase2_live_inference_pi5.py
   ```

3. **Use provided recovery script**
   ```bash
   bash recover_camera_ultimate.sh  # better than manual
   ```

4. **Check device before inference**
   ```bash
   python3 test_camera_minimal.py  # verify first
   ```

5. **One inference at a time**
   - Don't run multiple instances
   - Each uses entire camera device

---

## 📞 File Locations

All scripts in: `/Users/sittasahathum/Desktop/pilltrack-app/`

- `recover_camera_ultimate.sh` ⭐ Main script
- `release_camera_pi5.py` - Device release
- `test_camera_minimal.py` - Quick test
- `diagnose_camera.py` - Full diagnostic
- `CAMERA_TROUBLESHOOTING.md` - Extended guide
- `phase2_live_inference_pi5.py` - Main inference (optimized)
- `requirements_pi5.txt` - Dependencies
- `install_pi5_python313.sh` - Installation script

---

## 🎬 Next Steps

### ✅ On Pi5, run:

```bash
# Step 1: Auto recovery
bash recover_camera_ultimate.sh

# Or manual:
python3 release_camera_pi5.py
python3 test_camera_minimal.py
python3 phase2_live_inference_pi5.py
```

### ✅ Expected output:
```
📷 Starting inference...
✨ Processed 60 frames, 12 detections, FPS: 14.2
✨ Processed 120 frames, 25 detections, FPS: 14.1
...
```

### ✅ Press Ctrl+C to stop

---

## 📚 Related Documentation

- `CAMERA_TROUBLESHOOTING.md` - Extended troubleshooting
- `PI5_PERFORMANCE_OPTIMIZATION.md` - Performance tuning
- `FIXES_FOR_PI5_DEPLOYMENT.md` - Pre-deployment checklist
- `NUMPY_ERROR_FIXED.md` - Installation fixes

---

**Questions? Check the script comments or run `python3 diagnose_camera.py` for system details.**
