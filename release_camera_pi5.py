#!/usr/bin/env python3
"""
🎥 Release Camera - ปล่อยกล้องจากโปรแกรมที่ใช้งาน
"""

import subprocess
import sys
import time
import logging
import signal
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def kill_camera_processes():
    """ฆ่ากระบวนการที่ใช้กล้อง"""
    logger.info("=" * 60)
    logger.info("🔓 RELEASING CAMERA")
    logger.info("=" * 60)
    
    logger.info("\n🔍 Finding and killing processes using camera...")
    
    # โปรแกรมที่ต้อง kill
    processes = [
        ('python3', 'phase2_live_inference'),
        ('python3', 'phase1_database'),
        ('libcamera-hello', None),
        ('libcamera-still', None),
        ('raspivid', None),
        ('raspistill', None),
    ]
    
    for cmd, pattern in processes:
        try:
            if pattern:
                search_cmd = f"pkill -f '{pattern}'"
            else:
                search_cmd = f"pkill '{cmd}'"
            
            logger.info(f"   Killing: {search_cmd}")
            subprocess.run(search_cmd, shell=True, timeout=2)
            time.sleep(0.5)
        except:
            pass
    
    # Try more aggressive kill
    logger.info("\n   Forcing camera device reset...")
    try:
        subprocess.run("fuser -k /dev/video0", shell=True, timeout=2)
        logger.info("   ✅ Killed /dev/video0 users")
    except:
        pass
    
    # Unload camera module (more aggressive)
    logger.info("\n   Trying kernel module reset...")
    try:
        subprocess.run("sudo modprobe -r imx708", shell=True, timeout=2)
        logger.info("   ✅ Unloaded imx708 module")
        time.sleep(1)
        subprocess.run("sudo modprobe imx708", shell=True, timeout=2)
        logger.info("   ✅ Reloaded imx708 module")
    except:
        logger.warning("   ⚠️  Could not reset kernel module (need sudo)")
    
    logger.info("\n⏳ Waiting 3 seconds for camera to be released...")
    time.sleep(3)
    
    logger.info("✅ Camera release complete\n")
    logger.info("=" * 60)
    logger.info("📝 NEXT STEPS:")
    logger.info("=" * 60)
    logger.info("Option 1 - Auto reset and run:")
    logger.info("   bash reset_and_run_pi5.sh")
    logger.info("\nOption 2 - Manual retry:")
    logger.info("   python3 phase2_live_inference_pi5.py")
    logger.info("\nOption 3 - Full reboot:")
    logger.info("   sudo reboot")
    logger.info("=" * 60)

if __name__ == '__main__':
    kill_camera_processes()
