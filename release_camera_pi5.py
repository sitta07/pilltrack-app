#!/usr/bin/env python3
"""
🎥 Release Camera - ปล่อยกล้องจากโปรแกรมที่ใช้งาน
"""

import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def kill_camera_processes():
    """ฆ่ากระบวนการที่ใช้กล้อง"""
    logger.info("=" * 60)
    logger.info("🔓 RELEASING CAMERA")
    logger.info("=" * 60)
    
    # โปรแกรมที่อาจใช้กล้อง
    processes_to_check = [
        'libcamera',
        'libcamera-hello',
        'python3',
        'raspivid',
        'raspistill',
    ]
    
    logger.info("\n🔍 Finding processes using camera...")
    
    # ใช้ lsof เพื่อหาว่าใครใช้ /dev/video*
    try:
        result = subprocess.run(
            ["lsof", "/dev/video0"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.stdout:
            logger.info("📋 Processes using /dev/video0:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"   {line}")
            
            # Extract PIDs and kill them
            lines = result.stdout.split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            logger.info(f"   Killing PID {pid}...")
                            subprocess.run(["kill", "-9", str(pid)], timeout=2)
                            logger.info(f"   ✅ Killed PID {pid}")
                        except:
                            pass
    except FileNotFoundError:
        logger.warning("⚠️  lsof not available, trying pkill...")
        # Fallback: kill Python processes
        try:
            subprocess.run(["pkill", "-f", "phase2_live_inference"], timeout=2)
            logger.info("✅ Killed phase2_live_inference processes")
        except:
            pass
    except Exception as e:
        logger.warning(f"⚠️  Error checking processes: {e}")
    
    logger.info("\n⏳ Waiting 2 seconds for camera to be released...")
    time.sleep(2)
    
    logger.info("✅ Camera should now be available\n")
    logger.info("=" * 60)
    logger.info("📝 NEXT STEPS:")
    logger.info("=" * 60)
    logger.info("1. Retry the inference:")
    logger.info("   python3 phase2_live_inference_pi5.py")
    logger.info("\n2. If still fails, reboot:")
    logger.info("   sudo reboot")
    logger.info("=" * 60)

if __name__ == '__main__':
    kill_camera_processes()
