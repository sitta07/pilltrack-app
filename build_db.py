import os
import cv2
import pickle
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from ultralytics import YOLO
from engines import SIFTIdentifier, YOLODetector
import config
from PIL import Image

RAW_DATA_DIR = 'drug-scraping-c'
DEBUG_DIR = 'debug_crops'

class DrugFeatureExtractor(nn.Module):
    """Neural Network สำหรับสกัด features จากภาพยา"""
    def __init__(self, backbone='resnet18', feature_dim=256):
        super().__init__()
        
        # ใช้ pre-trained model เป็น backbone
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=True)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model
        elif backbone == 'efficientnet':
            base_model = models.efficientnet_b0(pretrained=True)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.backbone = base_model
        
        # Projection head สำหรับลด dimension
        self.projection = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim)
        )
        
        # Augmentation สำหรับ training
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        features = self.backbone(x)
        return self.projection(features)
    
    def extract_features(self, image):
        """Extract features จากภาพ"""
        self.eval()
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                # แปลงจาก OpenCV BGR เป็น RGB
                if len(image.shape) == 2:  # Grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            image = self.eval_transform(image).unsqueeze(0)
            features = self(image)
            return features.squeeze().numpy()

class NeuralDrugIdentifier:
    """Identifier ที่ใช้ Neural Network"""
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = DrugFeatureExtractor()
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.to(device)
        self.model.eval()
        
        # สำหรับรองรับ format เดิม
        self.sift_identifier = SIFTIdentifier()
    
    def extract_features(self, image):
        """Extract features โดยใช้ทั้ง Neural Network และ SIFT (เพื่อความเข้ากันได้)"""
        # 1. ใช้ Neural Network
        nn_features = self.model.extract_features(image)
        
        # 2. ใช้ SIFT (สำหรับ backward compatibility)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        kp, des = self.sift_identifier.extract_features(gray)
        
        return {
            'nn_features': nn_features,
            'sift_kp': kp,
            'sift_des': des
        }
    
    def generate_augmentations(self, image, num_augments=10):
        """สร้าง augmented images จากภาพเดียว สำหรับ few-shot learning"""
        augmented_images = []
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        for i in range(num_augments):
            # Random augmentations ที่หลากหลาย
            transform = transforms.Compose([
                transforms.RandomApply([
                    transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(45),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                ], p=0.8),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            augmented = transform(image)
            augmented_images.append(augmented)
        
        return augmented_images

def serialize_features(data):
    """Serialize features สำหรับเก็บลง database"""
    serialized = []
    
    for item in data:
        # เก็บทั้ง NN features และ SIFT features
        serialized_item = {
            'name': item['name'],
            'nn_features': item['nn_features'].tolist() if isinstance(item['nn_features'], np.ndarray) else item['nn_features'],
        }
        
        # เก็บ SIFT features ถ้ามี
        if 'sift_kp' in item:
            serialized_item['sift_kp'] = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) 
                                         for p in item['sift_kp']]
        if 'sift_des' in item and item['sift_des'] is not None:
            serialized_item['sift_des'] = item['sift_des'].tolist()
        
        serialized.append(serialized_item)
    
    return serialized

def main():
    print("🔨 Building Hybrid Database (Neural Network + SIFT)...")
    
    # เคลียร์โฟลเดอร์ Debug
    if os.path.exists(DEBUG_DIR):
        shutil.rmtree(DEBUG_DIR)
    os.makedirs(DEBUG_DIR)
    
    # 1. Load Models
    if config.DB_YOLO_PATH.endswith('.onnx'):
        model_path = config.DB_YOLO_PATH.replace('.onnx', '.pt')
    else:
        model_path = config.DB_YOLO_PATH
    
    print(f"🔹 Loading YOLO Model: {model_path}")
    yolo = YOLODetector(model_path, task='segment')
    
    # 2. Initialize Neural Network Identifier
    print("🔹 Initializing Neural Network Feature Extractor...")
    nn_identifier = NeuralDrugIdentifier()
    
    db_data = []
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ Error: Folder '{RAW_DATA_DIR}' not found!")
        return
    
    # 3. Process each drug
    for drug_name in os.listdir(RAW_DATA_DIR):
        drug_path = os.path.join(RAW_DATA_DIR, drug_name)
        if not os.path.isdir(drug_path):
            continue
        
        print(f"   💊 Processing: {drug_name}...")
        
        # รวบรวมทุกภาพของยาตัวนี้ (สำหรับ few-shot learning)
        drug_images = []
        for file_name in os.listdir(drug_path):
            if not file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            
            img_path = os.path.join(drug_path, file_name)
            image = cv2.imread(img_path)
            if image is not None:
                drug_images.append(image)
        
        # ถ้ามีหลายภาพ ใช้ few-shot learning
        if len(drug_images) > 0:
            # ใช้ภาพแรกเป็น base
            base_image = drug_images[0]
            
            # สร้าง augmentations สำหรับ training แบบ few-shot
            if len(drug_images) == 1:
                # ถ้ามีภาพเดียว สร้าง augmented images
                pil_image = Image.fromarray(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
                augmented = nn_identifier.model.train_transform(pil_image)
                drug_images_tensor = [augmented]
                
                # สร้าง augmented images เพิ่ม
                for _ in range(5):  # เพิ่มอีก 5 augmentations
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.2, 0.9)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(30),
                        transforms.ColorJitter(0.3, 0.3, 0.3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    augmented = transform(pil_image)
                    drug_images_tensor.append(augmented)
            
            # Detect และ crop จากทุกภาพ
            all_features = []
            for img_idx, frame in enumerate(drug_images):
                results = yolo.detect(frame, conf=0.5)
                
                for i, box in enumerate(results.boxes):
                    mask = results.masks[i] if results.masks else None
                    crop_img = yolo.get_crop(frame, box, mask)
                    
                    if crop_img is not None and crop_img.size > 0:
                        # Save Debug Image
                        debug_path = os.path.join(DEBUG_DIR, f"{drug_name}_{img_idx}_{i}.jpg")
                        cv2.imwrite(debug_path, crop_img)
                        
                        # สร้าง partial crops (จำลองการตัดครึ่งกล่อง)
                        h, w = crop_img.shape[:2]
                        partial_crops = [
                            crop_img,  # ภาพเต็ม
                            crop_img[0:h//2, :],  # ครึ่งบน
                            crop_img[h//2:, :],   # ครึ่งล่าง
                            crop_img[:, 0:w//2],  # ครึ่งซ้าย
                            crop_img[:, w//2:],   # ครึ่งขวา
                            crop_img[h//4:3*h//4, w//4:3*w//4]  # ตรงกลาง
                        ]
                        
                        # Extract features จากทุก partial crop
                        for partial_idx, partial_crop in enumerate(partial_crops):
                            if partial_crop.size == 0:
                                continue
                            
                            # Extract features ด้วย Neural Network
                            features_dict = nn_identifier.extract_features(partial_crop)
                            
                            # เก็บข้อมูล
                            db_data.append({
                                'name': drug_name,
                                'nn_features': features_dict['nn_features'],
                                'sift_kp': features_dict['sift_kp'],
                                'sift_des': features_dict['sift_des'],
                                'is_partial': partial_idx > 0,  # บอกว่าเป็น partial crop หรือไม่
                                'partial_type': partial_idx
                            })
    
    # 4. Save Database
    serialized_db = serialize_features(db_data)
    
    # บันทึกใน format เดิม (compatible)
    with open(config.DB_FILE_PATH, 'wb') as f:
        pickle.dump(serialized_db, f)
    
    # บันทึก neural network model แยกต่างหาก
    model_save_path = config.DB_FILE_PATH.replace('.pkl', '_nn_model.pt')
    torch.save(nn_identifier.model.state_dict(), model_save_path)
    
    print(f"🎉 Hybrid Database Built Successfully!")
    print(f"📁 Database saved to: {config.DB_FILE_PATH}")
    print(f"🧠 Neural Network Model saved to: {model_save_path}")
    print(f"📊 Total entries: {len(db_data)}")

if __name__ == "__main__":
    main()