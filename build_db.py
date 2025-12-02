import os
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from engines import SIFTIdentifier, YOLODetector
import config

RAW_DATA_DIR = 'drug-scraping-c'

def serialize_keypoints(kp):
    return [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp]

def main():
    print("🔨 Building Database (Legacy Logic: Grayscale Only)...")
    
    yolo = YOLODetector(config.DB_YOLO_PATH) # ใช้ Class ที่แก้แล้ว
    identifier = SIFTIdentifier()               # ใช้ Class ที่แก้แล้ว
    
    db_data = []
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ Error: Folder '{RAW_DATA_DIR}' not found!")
        return

    for drug_name in os.listdir(RAW_DATA_DIR):
        drug_path = os.path.join(RAW_DATA_DIR, drug_name)
        if not os.path.isdir(drug_path): continue
        
        print(f"   💊 Processing: {drug_name}...")
        
        for file_name in os.listdir(drug_path):
            if not file_name.lower().endswith(('.jpg', '.png', '.jpeg')): continue
            
            img_path = os.path.join(drug_path, file_name)
            frame = cv2.imread(img_path)
            if frame is None: continue

            # 1. Detect
            results = yolo.detect(frame,conf=0.5)
            
            for i, box in enumerate(results.boxes):
                mask = results.masks[i] if results.masks else None
                
                # 2. Crop (ใช้ Logic เดิมเป๊ะๆ จาก engines.py)
                crop_img = yolo.get_crop(frame, box, mask)
                
                # 3. Extract Features (Grayscale + CLAHE)
                kp, des = identifier.extract_features(crop_img)
                
                # เก็บเฉพาะภาพที่มีจุดเด่นมากพอ (>15 จุด) ตามโค้ดเก่า
                if des is not None and len(kp) > 10:
                    db_data.append({
                        'name': drug_name,
                        'kp': serialize_keypoints(kp),
                        'des': des
                        # ❌ ตัด 'hist' ออก ไม่เก็บเรื่องสี
                    })

    print(f"✅ Saving {len(db_data)} entries to {config.DB_FILE_PATH}")
    with open(config.DB_FILE_PATH, 'wb') as f:
        pickle.dump(db_data, f)
    print("🎉 Database Build Complete!")

if __name__ == "__main__":
    main()