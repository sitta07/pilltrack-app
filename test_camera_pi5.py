#!/usr/bin/env python3
"""
🎥 Simple Camera Test for Raspberry Pi 5
ตรวจสอบว่ากล้องทำงาน
"""

import cv2
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_opencv_camera():
    """Test OpenCV camera access"""
    logger.info("=" * 60)
    logger.info("🔧 TEST 1: OpenCV Camera Access")
    logger.info("=" * 60)
    
    try:
        logger.info("📷 Attempting to open camera with OpenCV...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("❌ Cannot open camera with VideoCapture(0)")
            return False
        
        logger.info("✅ VideoCapture(0) opened successfully")
        
        # Try to read frame
        logger.info("📸 Attempting to read first frame...")
        ret, frame = cap.read()
        
        if not ret or frame is None:
            logger.error("❌ Cannot read frame from camera")
            cap.release()
            return False
        
        logger.info(f"✅ Frame read successfully: {frame.shape}")
        
        # Read a few more frames
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                logger.info(f"   Frame {i+2}: {frame.shape}")
            else:
                logger.warning(f"   Frame {i+2}: Failed to read")
        
        cap.release()
        logger.info("✅ OpenCV Camera Test PASSED\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenCV Camera Test FAILED: {e}\n")
        return False

def test_picamera2():
    """Test Picamera2 access"""
    logger.info("=" * 60)
    logger.info("🔧 TEST 2: Picamera2 Access")
    logger.info("=" * 60)
    
    try:
        logger.info("📷 Importing Picamera2...")
        from picamera2 import Picamera2
        logger.info("✅ Picamera2 imported")
        
        logger.info("📷 Creating Picamera2 object...")
        cam = Picamera2(0)
        logger.info("✅ Picamera2(0) created")
        
        logger.info("⚙️  Creating configuration...")
        config = cam.create_preview_configuration(main={"size": (640, 480)})
        logger.info("✅ Configuration created")
        
        logger.info("⚙️  Applying configuration...")
        cam.configure(config)
        logger.info("✅ Configuration applied")
        
        logger.info("🎬 Starting camera...")
        cam.start()
        logger.info("✅ Camera started")
        
        logger.info("📸 Attempting to capture frames...")
        for i in range(3):
            frame = cam.capture_array()
            if frame is not None and frame.size > 0:
                logger.info(f"   Frame {i+1}: {frame.shape}")
            else:
                logger.warning(f"   Frame {i+1}: Empty or None")
        
        cam.stop()
        logger.info("✅ Picamera2 Test PASSED\n")
        return True
        
    except ImportError:
        logger.warning("⚠️  Picamera2 not installed")
        logger.info("   Install with: pip install picamera2\n")
        return False
    except Exception as e:
        logger.error(f"❌ Picamera2 Test FAILED: {e}\n")
        return False

def test_libcamera():
    """Test libcamera command"""
    logger.info("=" * 60)
    logger.info("🔧 TEST 3: libcamera (System)")
    logger.info("=" * 60)
    
    try:
        import subprocess
        
        logger.info("🔍 Checking libcamera with command line...")
        result = subprocess.run(
            ["libcamera-hello", "--list-cameras"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            logger.info("✅ libcamera found cameras:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"   {line}")
            return True
        else:
            logger.warning("⚠️  libcamera-hello not available or no cameras found")
            return False
            
    except FileNotFoundError:
        logger.warning("⚠️  libcamera-hello command not found")
        return False
    except Exception as e:
        logger.warning(f"⚠️  libcamera test failed: {e}")
        return False

def main():
    logger.info("\n" + "=" * 60)
    logger.info("🎥 RASPBERRY PI 5 CAMERA TEST SUITE")
    logger.info("=" * 60 + "\n")
    
    results = []
    
    # Test 3: libcamera (system)
    results.append(("libcamera (System)", test_libcamera()))
    
    # Test 2: Picamera2
    results.append(("Picamera2", test_picamera2()))
    
    # Test 1: OpenCV (most important)
    results.append(("OpenCV", test_opencv_camera()))
    
    # Summary
    logger.info("=" * 60)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name:30} {status}")
    
    logger.info("=" * 60)
    
    # Recommendation
    logger.info("\n🎯 RECOMMENDATIONS:")
    
    if results[2][1]:  # OpenCV passed
        logger.info("✅ OpenCV camera works! Use it for inference.")
        logger.info("   Run: python3 phase2_live_inference_pi5.py")
    elif results[1][1]:  # Picamera2 passed
        logger.info("✅ Picamera2 works! Will use it automatically.")
        logger.info("   Run: python3 phase2_live_inference_pi5.py")
    elif results[0][1]:  # libcamera passed
        logger.info("⚠️  Camera is detected by system but Python interface failed.")
        logger.info("   Try: pip install --upgrade picamera2")
    else:
        logger.error("❌ No working camera interface found!")
        logger.error("   Check:")
        logger.error("   1. Camera is connected and enabled")
        logger.error("   2. Run: raspi-config → Interface Options → Camera → Enable")
        logger.error("   3. Reboot and try again")

if __name__ == '__main__':
    main()
