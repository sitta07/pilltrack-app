#!/usr/bin/env python3
"""
🔬 Camera Diagnostic Tool
ตรวจสอบทุกชิ้นส่วนของระบบกล้อง
"""

import subprocess
import sys
import os
import time
import cv2

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_section(title):
    print(f"\n✓ {title}")
    print("-" * 70)

def run_cmd(cmd, description=""):
    """รันคำสั่ง bash และแสดงผล"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.stdout.strip():
            print(f"  {description}")
            for line in result.stdout.strip().split('\n'):
                print(f"    {line}")
        if result.returncode != 0 and result.stderr.strip():
            print(f"  ⚠️  {result.stderr.strip()[:200]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ⏱️  Command timed out: {cmd}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {str(e)[:100]}")
        return False

def test_system_info():
    """ตรวจสอบข้อมูลระบบ"""
    print_section("System Information")
    
    # Check OS
    run_cmd("uname -a", "OS Info:")
    run_cmd("cat /etc/os-release | head -3", "OS Release:")
    
    # Check Python
    run_cmd("python3 --version", "Python Version:")
    
    # Check if on Raspberry Pi
    if os.path.exists('/proc/cpuinfo'):
        run_cmd("cat /proc/cpuinfo | grep -E 'Model|Hardware'", "CPU Model:")

def test_kernel_modules():
    """ตรวจสอบ kernel modules"""
    print_section("Kernel Modules")
    
    run_cmd("lsmod | grep -E 'imx|video|uvc'", "Camera-related modules:")
    run_cmd("modinfo imx708 2>/dev/null | head -5", "IMX708 driver info:")

def test_camera_devices():
    """ตรวจสอบ camera devices"""
    print_section("Camera Devices")
    
    # List video devices
    run_cmd("ls -la /dev/video* 2>/dev/null", "Video devices:")
    
    # Check device permissions
    run_cmd("ls -la /dev/video0", "Video0 permissions:")
    
    # Check if device is busy
    print("\n  Checking for processes using camera:")
    run_cmd("fuser /dev/video0 2>/dev/null", "  Processes:")
    if not run_cmd("lsof /dev/video0 2>/dev/null", "  Detailed info:"):
        print("    (No processes found - camera should be free)")

def test_libcamera():
    """ตรวจสอบ libcamera"""
    print_section("libcamera Support")
    
    # Check if libcamera is installed
    if run_cmd("which libcamera-hello", "libcamera-hello location:"):
        print("\n  Testing libcamera-hello (3 second timeout):")
        start = time.time()
        try:
            result = subprocess.run(
                "timeout 3 libcamera-hello --timeout 2000 2>&1 | head -10",
                shell=True,
                capture_output=True,
                text=True
            )
            elapsed = time.time() - start
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    print(f"    {line}")
            print(f"    ⏱️  Completed in {elapsed:.1f} seconds")
        except Exception as e:
            print(f"    ❌ Error: {str(e)[:100]}")
    else:
        print("  ❌ libcamera-hello not found")
        print("    Install: sudo apt install -y libcamera-tools")

def test_opencv():
    """ตรวจสอบ OpenCV"""
    print_section("OpenCV Support")
    
    try:
        print(f"  OpenCV version: {cv2.__version__}")
        print(f"  OpenCV build info:")
        print(f"    - Video backends: {cv2.getBuildInformation().split('Video I/O:')[1].split('\\n')[0] if 'Video I/O:' in cv2.getBuildInformation() else 'Unknown'}")
    except Exception as e:
        print(f"  ⚠️  Could not get OpenCV info: {str(e)[:100]}")

def test_opencv_camera():
    """ตรวจสอบ camera ผ่าน OpenCV"""
    print_section("OpenCV Camera Test")
    
    print("  Opening camera with cv2.VideoCapture(0)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("  ❌ Could not open camera!")
        print("    - Device may be busy")
        print("    - Run: python3 release_camera_pi5.py")
        return False
    
    print("  ✅ Camera opened successfully")
    
    # Try to get properties
    print("\n  Camera Properties:")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"    - Resolution: {width}x{height}")
    print(f"    - FPS: {fps}")
    print(f"    - Backend: {cap.getBackendName() if hasattr(cap, 'getBackendName') else 'Unknown'}")
    
    # Try to read a frame
    print("\n  Reading frame...")
    ret, frame = cap.read()
    
    if ret:
        print(f"  ✅ Frame captured successfully!")
        print(f"    - Frame shape: {frame.shape}")
        print(f"    - Data type: {frame.dtype}")
        cap.release()
        return True
    else:
        print("  ❌ Could not read frame from camera")
        cap.release()
        return False

def test_picamera2():
    """ตรวจสอบ Picamera2"""
    print_section("Picamera2 Support")
    
    try:
        from picamera2 import Picamera2
        print("  ✅ Picamera2 imported successfully")
        
        print("\n  Attempting to create Picamera2 instance...")
        pic2 = Picamera2()
        print("  ✅ Picamera2 instance created")
        
        print("\n  Available cameras:")
        from picamera2.utils import get_cameras
        cameras = get_cameras()
        print(f"    Found {len(cameras)} camera(s)")
        
        pic2.close()
        return True
    except ImportError:
        print("  ⚠️  Picamera2 not installed")
        print("    Install: sudo apt install -y -q python3-picamera2")
        return False
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

def test_tensorflow():
    """ตรวจสอบ TensorFlow/PyTorch"""
    print_section("ML Framework Support")
    
    try:
        import torch
        print(f"  ✅ PyTorch version: {torch.__version__}")
        print(f"    - CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("  ⚠️  PyTorch not installed")
    
    try:
        import tensorflow
        print(f"  ✅ TensorFlow version: {tensorflow.__version__}")
    except ImportError:
        print("  ⚠️  TensorFlow not installed")

def main():
    print_header("🔬 Pi5 Camera Diagnostic Report")
    print(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    test_system_info()
    test_kernel_modules()
    test_camera_devices()
    test_libcamera()
    test_opencv()
    test_picamera2()
    test_tensorflow()
    
    # Critical test - can we actually use the camera?
    opencv_success = test_opencv_camera()
    
    # Summary
    print_header("📋 Summary")
    if opencv_success:
        print("  ✅ Camera is working!")
        print("\n  Ready to run:")
        print("    python3 phase2_live_inference_pi5.py")
    else:
        print("  ❌ Camera is not working!")
        print("\n  Solutions to try:")
        print("    1. python3 release_camera_pi5.py")
        print("    2. sudo reboot")
        print("    3. sudo fuser -k /dev/video0")
    
    print("\n" + "=" * 70 + "\n")

if __name__ == '__main__':
    main()
