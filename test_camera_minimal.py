#!/usr/bin/env python3
"""
⚡ Minimal Camera Test - ใช้สำหรับ debug อย่างรวดเร็ว
"""

import subprocess
import time
import sys

def main():
    print("\n" + "=" * 60)
    print("⚡ MINIMAL CAMERA TEST")
    print("=" * 60)
    
    # Step 1: Kill processes
    print("\n🔓 Killing processes...")
    subprocess.run("pkill -f phase2_live_inference || true", shell=True)
    subprocess.run("pkill -f libcamera || true", shell=True)
    time.sleep(1)
    
    # Step 2: Test with libcamera first
    if subprocess.run("which libcamera-hello > /dev/null 2>&1", shell=True).returncode == 0:
        print("\n📷 Testing with libcamera...")
        result = subprocess.run(
            "timeout 2 libcamera-hello --timeout 1000 2>&1 | head -5",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
    
    # Step 3: Test with OpenCV
    print("\n🐍 Testing with OpenCV...")
    test_code = """
import cv2
import sys
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f'✅ SUCCESS: {frame.shape}')
        cap.release()
        sys.exit(0)
    else:
        print('❌ Could not read frame')
        cap.release()
        sys.exit(1)
else:
    print('❌ Could not open camera')
    sys.exit(1)
"""
    
    result = subprocess.run(f"python3 -c \"{test_code}\"", shell=True)
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✅ CAMERA IS WORKING!")
        print("=" * 60)
        print("\nRun inference:")
        print("  python3 phase2_live_inference_pi5.py")
        print("\n" + "=" * 60 + "\n")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ CAMERA FAILED!")
        print("=" * 60)
        print("\nTry:")
        print("  1. python3 release_camera_pi5.py")
        print("  2. python3 test_camera_minimal.py  (retry)")
        print("  3. sudo reboot")
        print("\n" + "=" * 60 + "\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
