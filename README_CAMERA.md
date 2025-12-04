# 🚀 Pi5 Camera - Quick Start

## ⚡ 30-Second Solution

If you're getting **"Device or resource busy"** error:

```bash
bash recover_camera_ultimate.sh
```

That's it! This script will:
1. Kill all stuck processes
2. Release camera device  
3. Test camera
4. Start inference

---

## 📍 Common Scenarios

### Scenario 1: Fresh Start
```bash
python3 phase2_live_inference_pi5.py
```

### Scenario 2: Camera Error
```bash
bash recover_camera_ultimate.sh
```

### Scenario 3: Quick Test
```bash
python3 test_camera_minimal.py
```

### Scenario 4: Debug Everything
```bash
python3 diagnose_camera.py
```

### Scenario 5: Full Reboot
```bash
sudo reboot
```

---

## ✅ Verify Setup

Before running, check:

```bash
# Camera device exists
ls /dev/video0

# No stuck processes
ps aux | grep python

# Camera works
python3 test_camera_minimal.py
```

---

## 📊 Expected Output

```
📷 Starting inference...
✨ Processed 60 frames, 12 detections, FPS: 14.2
✨ Processed 120 frames, 25 detections, FPS: 14.1
✨ Processed 180 frames, 38 detections, FPS: 14.0

Model Detection: ['Medicine A', 'Medicine B', 'Medicine C']
Confidence: 0.87
```

---

## 🎯 Stop Running

Press `Ctrl+C` to stop inference gracefully.

---

## 📚 More Help

- `CAMERA_RECOVERY_GUIDE.md` - Detailed troubleshooting
- `CAMERA_TROUBLESHOOTING.md` - Error explanations
- `diagnose_camera.py` - System diagnostic

---

## 🆘 Still Having Issues?

1. **Try recovery script**: `bash recover_camera_ultimate.sh --aggressive`
2. **Full reboot**: `sudo reboot`
3. **Check hardware**: Reseat camera ribbon cable
4. **Get diagnostics**: `python3 diagnose_camera.py`

---

**Ready? Run: `bash recover_camera_ultimate.sh`**
