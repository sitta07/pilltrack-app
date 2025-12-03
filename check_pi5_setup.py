#!/usr/bin/env python3
"""
PillTrack - Raspberry Pi 5 Setup Verification
ตรวจสอบว่า PillTrack พร้อมสำหรับ Pi 5
"""

import os
import sys
from pathlib import Path

def check_file_exists(filename):
    """ตรวจสอบไฟล์"""
    exists = os.path.isfile(filename)
    symbol = "✅" if exists else "❌"
    size = ""
    if exists:
        size = f" ({os.path.getsize(filename) / 1024:.1f} KB)"
    print(f"  {symbol} {filename}{size}")
    return exists

def check_directory(dirname):
    """ตรวจสอบโฟลเดอร์"""
    exists = os.path.isdir(dirname)
    symbol = "✅" if exists else "❌"
    print(f"  {symbol} {dirname}")
    return exists

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║      🔧 PillTrack Raspberry Pi 5 Setup Verification           ║")
    print("║           ตรวจสอบว่าทั้งหมดพร้อมสำหรับ Pi 5                   ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    all_ok = True
    
    # 1. ตรวจสอบไฟล์ Pi 5
    print("📄 Pi 5 Files (ตรวจสอบ):")
    pi5_files = [
        "phase1_database_preparation_pi5.py",
        "phase2_live_inference_pi5.py",
        "requirements_pi5.txt",
        "setup_pi5.sh",
        "README_PI5.md",
        "PILLTRACK_PI5_CHANGES.md",
        "FILES_INDEX_TH.md",
    ]
    for f in pi5_files:
        if not check_file_exists(f):
            all_ok = False
    print()
    
    # 2. ตรวจสอบ Documentation
    print("📚 Documentation (ตรวจสอบ):")
    doc_files = [
        "SYSTEM_ARCHITECTURE.md",
        "IMPLEMENTATION_GUIDE.md",
        "COMPLETE_SUMMARY.md",
    ]
    for f in doc_files:
        if not check_file_exists(f):
            all_ok = False
    print()
    
    # 3. ตรวจสอบ Models
    print("🤖 Models (ตรวจสอบ):")
    models = [
        "best_process_2.onnx",
        "best_process_2.pt",
        "seg_db_best.pt",
    ]
    for m in models:
        if not check_file_exists(m):
            all_ok = False
    print()
    
    # 4. ตรวจสอบ Data Folder
    print("📁 Data Folders (ตรวจสอบ):")
    folders = [
        "drug-scraping-c",
        "data",
    ]
    for f in folders:
        if not check_directory(f):
            all_ok = False
    print()
    
    # 5. สรุป
    print("════════════════════════════════════════════════════════════════")
    if all_ok:
        print("✅ ทั้งหมดพร้อม! PillTrack สำหรับ Pi 5 เสร็จสิ้น")
        print()
        print("📖 ขั้นตอนถัดไป:")
        print("  1. อ่าน README_PI5.md เพื่อรายละเอียด")
        print("  2. รัน: bash setup_pi5.sh")
        print("  3. รัน: python3 phase1_database_preparation_pi5.py")
        print("  4. รัน: python3 phase2_live_inference_pi5.py")
        print()
    else:
        print("⚠️ ไฟล์บางไฟล์ขาดหายไป ตรวจสอบการสร้าง")
        sys.exit(1)
    
    # 6. สรุปไฟล์ที่เพิ่มเข้ามา
    print("════════════════════════════════════════════════════════════════")
    print("📊 สรุปไฟล์ที่สร้างใหม่:")
    print()
    print("🚀 Pi 5 Ready:")
    print("   • phase1_database_preparation_pi5.py     (สร้าง database)")
    print("   • phase2_live_inference_pi5.py           (รันระบบจริง)")
    print("   • requirements_pi5.txt                   (dependencies)")
    print("   • setup_pi5.sh                           (auto setup)")
    print()
    print("📖 Guides:")
    print("   • README_PI5.md                          (ภาษาไทย)")
    print("   • FILES_INDEX_TH.md                      (ดัชนี)")
    print("   • PILLTRACK_PI5_CHANGES.md              (บันทึก)")
    print()
    print("════════════════════════════════════════════════════════════════")
    print()
    print("⚡ Performance Targets:")
    print("   • FPS: 12-15 (บน Pi 5)")
    print("   • Accuracy: 98%+")
    print("   • Latency: 100-150ms")
    print()
    print("✨ ใช้งาน:")
    print("   $ python3 phase2_live_inference_pi5.py")
    print("   🎥 กด 'q' เพื่อออก")
    print()
    print("🎉 พร้อมใช้งาน PillTrack บน Raspberry Pi 5!")
    print("════════════════════════════════════════════════════════════════")

if __name__ == '__main__':
    main()
