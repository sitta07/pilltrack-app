#!/usr/bin/env python3
"""
🤖 Model Check - ตรวจสอบว่าโมเดลทั้งหมดสามารถโหลดได้
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_files():
    """ตรวจสอบว่าไฟล์โมเดลมีอยู่"""
    logger.info("=" * 60)
    logger.info("📋 CHECKING MODEL FILES")
    logger.info("=" * 60)
    
    files_to_check = [
        'best_process_2.onnx',
        'seg_db_best.pt',
        'drug-scraping-c',  # Directory
        'faiss_database/drug_index.faiss',
        'faiss_database/metadata.json',
    ]
    
    all_exist = True
    for filepath in files_to_check:
        if os.path.exists(filepath):
            if os.path.isdir(filepath):
                count = len(os.listdir(filepath))
                logger.info(f"✅ {filepath:40} (folder with {count} items)")
            else:
                size = os.path.getsize(filepath) / 1024 / 1024
                logger.info(f"✅ {filepath:40} ({size:.1f} MB)")
        else:
            logger.error(f"❌ {filepath:40} NOT FOUND!")
            all_exist = False
    
    return all_exist

def check_yolo():
    """ตรวจสอบ YOLO"""
    logger.info("\n" + "=" * 60)
    logger.info("🤖 CHECKING YOLO MODEL")
    logger.info("=" * 60)
    
    try:
        logger.info("📦 Importing ultralytics...")
        from ultralytics import YOLO
        logger.info("✅ ultralytics imported")
        
        logger.info("📦 Loading YOLO model: best_process_2.onnx...")
        model = YOLO('best_process_2.onnx')
        logger.info(f"✅ YOLO model loaded: {model}")
        
        return True
    except Exception as e:
        logger.error(f"❌ YOLO failed: {e}")
        return False

def check_feature_extractor():
    """ตรวจสอบ Feature Extractor"""
    logger.info("\n" + "=" * 60)
    logger.info("🧠 CHECKING FEATURE EXTRACTOR (EfficientNet-B3)")
    logger.info("=" * 60)
    
    try:
        logger.info("📦 Importing timm...")
        import timm
        logger.info("✅ timm imported")
        
        logger.info("📦 Creating EfficientNet-B3 model...")
        logger.info("   (This may download from Hugging Face on first run)")
        model = timm.create_model('efficientnet_b3', pretrained=True)
        logger.info(f"✅ Feature extractor loaded")
        
        # Check model parameters
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Parameters: {params / 1e6:.1f}M")
        
        return True
    except Exception as e:
        logger.error(f"❌ Feature extractor failed: {e}")
        return False

def check_faiss():
    """ตรวจสอบ FAISS"""
    logger.info("\n" + "=" * 60)
    logger.info("📊 CHECKING FAISS DATABASE")
    logger.info("=" * 60)
    
    try:
        logger.info("📦 Importing faiss...")
        import faiss
        logger.info("✅ faiss imported")
        
        if not os.path.exists('faiss_database/drug_index.faiss'):
            logger.warning("⚠️  Database not found. Run Phase 1 first:")
            logger.warning("   python3 phase1_database_preparation_pi5.py")
            return False
        
        logger.info("📦 Loading FAISS index...")
        index = faiss.read_index('faiss_database/drug_index.faiss')
        logger.info(f"✅ FAISS index loaded: {index.ntotal} drugs")
        
        logger.info("📦 Loading metadata...")
        import json
        with open('faiss_database/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"✅ Metadata loaded: {len(metadata['drug_names'])} drugs")
        logger.info(f"   Sample drugs: {metadata['drug_names'][:3]}")
        
        return True
    except Exception as e:
        logger.error(f"❌ FAISS failed: {e}")
        return False

def check_imports():
    """ตรวจสอบ imports ที่จำเป็น"""
    logger.info("\n" + "=" * 60)
    logger.info("📦 CHECKING IMPORTS")
    logger.info("=" * 60)
    
    imports = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('albumentations', 'Albumentations'),
    ]
    
    all_ok = True
    for module_name, display_name in imports:
        try:
            __import__(module_name)
            logger.info(f"✅ {display_name:20} ({module_name})")
        except ImportError:
            logger.error(f"❌ {display_name:20} ({module_name}) NOT INSTALLED")
            all_ok = False
    
    return all_ok

def main():
    logger.info("\n" + "=" * 60)
    logger.info("🔍 MODEL AND DEPENDENCY CHECK")
    logger.info("=" * 60 + "\n")
    
    results = []
    
    # Check imports first
    results.append(("Python Imports", check_imports()))
    
    # Check files
    results.append(("Model Files", check_files()))
    
    # Check models
    results.append(("YOLO Model", check_yolo()))
    results.append(("Feature Extractor", check_feature_extractor()))
    results.append(("FAISS Database", check_faiss()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 SUMMARY")
    logger.info("=" * 60)
    
    for check_name, passed in results:
        status = "✅ PASS" if passed else "⚠️  FAIL/WARN"
        logger.info(f"{check_name:30} {status}")
    
    logger.info("=" * 60)
    
    # Recommendation
    all_pass = all(result[1] for result in results)
    
    if all_pass:
        logger.info("\n✅ All checks passed!")
        logger.info("   Run: python3 phase2_live_inference_pi5.py")
    else:
        logger.error("\n❌ Some checks failed. Fix the issues above first.")
        logger.error("\nCommon fixes:")
        logger.error("1. Database not found → Run phase1_database_preparation_pi5.py")
        logger.error("2. Missing models → Check they exist in current folder")
        logger.error("3. Import errors → Run: pip install -r requirements_pi5.txt")

if __name__ == '__main__':
    main()
