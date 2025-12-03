import os
import cv2
import pickle
import numpy as np
import shutil
from ultralytics import YOLO
from engines import SIFTIdentifier, YOLODetector
import config

RAW_DATA_DIR = 'drug-scraping-c'
DEBUG_DIR = 'debug_crops' 

def serialize_keypoints(kp):
    return [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp]

def main():
    print("🔨 Building Database (Legacy Mode: Grayscale SIFT Only)...")
    
    # เคลียร์โฟลเดอร์ Debug
    if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
    os.makedirs(DEBUG_DIR)

    # 1. Load Model
    # ใช้ .pt เพื่อความชัวร์ (แก้ Path อัตโนมัติถ้า config ชี้ไป onnx)
    if config.DB_YOLO_PATH.endswith('.onnx'):
        model_path = config.DB_YOLO_PATH.replace('.onnx', '.pt')
    else:
        model_path = config.DB_YOLO_PATH
        
    print(f"🔹 Loading Model: {model_path}")
    # ใส่ task='segment' เพื่อกัน Warning
    yolo = YOLODetector(model_path, task='segment')
    
    identifier = SIFTIdentifier()
    db_data = []
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ Error: Folder '{RAW_DATA_DIR}' not found!")
        return

    # 2. Loop Processing
    for drug_name in os.listdir(RAW_DATA_DIR):
        drug_path = os.path.join(RAW_DATA_DIR, drug_name)
        if not os.path.isdir(drug_path): continue
        
        print(f"   💊 Processing: {drug_name}...")
        
        for idx, file_name in enumerate(os.listdir(drug_path)):
            if not file_name.lower().endswith(('.jpg', '.png', '.jpeg')): continue
            
            img_path = os.path.join(drug_path, file_name)
            frame = cv2.imread(img_path)
            if frame is None: continue

            # Detect
            results = yolo.detect(frame, conf=0.5)
            
            for i, box in enumerate(results.boxes):
                mask = results.masks[i] if results.masks else None
                
                # Crop
                crop_img = yolo.get_crop(frame, box, mask)
                
                if crop_img is not None and crop_img.size > 0:
                    # Save Debug Image
                    cv2.imwrite(os.path.join(DEBUG_DIR, f"{drug_name}_{idx}_{i}.jpg"), crop_img)

                    # Extract SIFT (Engine จะแปลงเป็น Grayscale เอง)
                    # รับแค่ kp, des (2 ค่า)
                    kp, des = identifier.extract_features(crop_img)
                    
                    if des is not None and len(kp) > 10:
                        db_data.append({
                            'name': drug_name,
                            'kp': serialize_keypoints(kp),
                            'des': des
                            # ❌ ไม่เก็บ 'hist'
                            # ❌ ไม่เก็บ 'type'
                        })

    # 3. Save
    with open(config.DB_FILE_PATH, 'wb') as f:
        pickle.dump(db_data, f)
    print("🎉 Legacy Database Built Successfully!")

if __name__ == "__main__":
    main()