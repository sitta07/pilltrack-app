import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image
import config

class DrugFeatureExtractor(nn.Module):
    """Neural Network สำหรับสกัด features จากภาพยา"""
    def __init__(self, backbone='resnet18', feature_dim=256):
        super().__init__()
        
        # ใช้ pre-trained model ที่เบาและเร็วสำหรับ Raspberry Pi
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=True)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model
        elif backbone == 'mobilenet_v2':
            base_model = models.mobilenet_v2(pretrained=True)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.backbone = base_model
        else:
            # default เป็น resnet18
            base_model = models.resnet18(pretrained=True)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model
        
        # Projection head
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
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
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
    
    def extract_features_numpy(self, image):
        """Extract features จาก numpy array (OpenCV image)"""
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
            return features.squeeze().cpu().numpy()
    
    def extract_features_batch(self, images):
        """Extract features จากหลายภาพ"""
        self.eval()
        with torch.no_grad():
            transformed_images = []
            for img in images:
                if isinstance(img, np.ndarray):
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                img = self.eval_transform(img)
                transformed_images.append(img)
            
            batch = torch.stack(transformed_images)
            features = self(batch)
            return features.cpu().numpy()

class NeuralDrugIdentifier:
    """Identifier ที่ใช้ Neural Network"""
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        
        # ใช้ backbone ที่เบาเพื่อความเร็วบน Raspberry Pi
        self.model = DrugFeatureExtractor(
            backbone=config.NEURAL_BACKBONE,
            feature_dim=config.NEURAL_FEATURE_DIM
        )
        
        if model_path and os.path.exists(model_path):
            try:
                # โหลดเฉพาะ state dict
                state_dict = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state_dict)
                print(f"✅ Loaded Neural Network from {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load neural model: {e}, using pretrained weights")
        
        self.model.to(device)
        self.model.eval()
        
        # สร้าง partial crops สำหรับ augment ภาพเดียว
        self.partial_crop_types = ['full', 'top', 'bottom', 'left', 'right', 'center']
    
    def extract_features(self, image):
        """Extract features โดยใช้ Neural Network"""
        try:
            features = self.model.extract_features_numpy(image)
            return features
        except Exception as e:
            print(f"❌ Error extracting neural features: {e}")
            return None
    
    def create_partial_crops(self, image):
        """สร้าง partial crops จากภาพ (สำหรับจำลองการเห็นบางส่วนของยา)"""
        h, w = image.shape[:2]
        partial_crops = []
        
        # ภาพเต็ม
        partial_crops.append(image)
        
        # ครึ่งบน
        if h >= 20:
            partial_crops.append(image[0:h//2, :])
        
        # ครึ่งล่าง
        if h >= 20:
            partial_crops.append(image[h//2:, :])
        
        # ครึ่งซ้าย
        if w >= 20:
            partial_crops.append(image[:, 0:w//2])
        
        # ครึ่งขวา
        if w >= 20:
            partial_crops.append(image[:, w//2:])
        
        # ตรงกลาง (60% ของภาพ)
        if h >= 60 and w >= 60:
            h_margin = int(h * 0.2)
            w_margin = int(w * 0.2)
            partial_crops.append(image[h_margin:h-h_margin, w_margin:w-w_margin])
        
        return partial_crops
    
    def compute_similarity(self, feat1, feat2):
        """คำนวณ similarity ระหว่างสอง features"""
        if feat1 is None or feat2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

class YOLODetector:
    def __init__(self, model_path, task=None):
        print(f"🔄 Loading YOLO Model: {model_path}...")
        self.model = YOLO(model_path, task=task)
    
    def detect(self, frame, conf=0.25, iou=0.25, agnostic_nms=True, max_det=100, **kwargs):
        results = self.model(frame, 
                             verbose=False, 
                             conf=conf, 
                             iou=iou, 
                             agnostic_nms=agnostic_nms, 
                             max_det=max_det,           
                             retina_masks=True,
                             **kwargs)
        return results[0]

    def get_crop(self, img, box, mask_data):
        h_orig, w_orig = img.shape[:2]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)
        
        if mask_data is None:
            return img[y1:y2, x1:x2]

        mask = mask_data.data[0].cpu().numpy()
        mask = cv2.resize(mask, (w_orig, h_orig))
        mask = (mask > 0.5).astype(np.uint8)
        
        bg = np.full_like(img, 0, dtype=np.uint8)
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        masked_img = np.where(mask_3ch == 1, img, bg)
        
        return masked_img[y1:y2, x1:x2]

class SIFTIdentifier:
    def __init__(self):
        print("⏳ Initializing SIFT Detector (Legacy: Grayscale + CLAHE)...")
        self.sift = cv2.SIFT_create(nfeatures=2000)
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def extract_features(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0: 
            return [], None
        
        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr
            
        gray = self.clahe.apply(gray)
        kp, des = self.sift.detectAndCompute(gray, None)
        
        return kp, des

    def compare(self, query_pack, db_pack):
        kp_q, des_q = query_pack
        kp_db, des_db = db_pack
        
        if des_q is None or des_db is None or len(des_q) < 2 or len(des_db) < 2:
            return 0
            
        try:
            matches = self.flann.knnMatch(des_q, des_db, k=2)
        except: 
            return 0

        good_matches = []
        for m, n in matches:
            if m.distance < config.SIFT_MATCH_RATIO * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 4: 
            return 0
            
        src_pts = np.float32([kp_q[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_db[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None: 
            return 0
        
        return mask.ravel().tolist().count(1)

class HybridMatcher:
    """Matcher ที่ใช้ทั้ง Neural Network และ SIFT"""
    def __init__(self, db_path, nn_model_path=None):
        import pickle
        import os
        
        print(f"🔍 Loading Hybrid Database from {db_path}...")
        
        # โหลด database
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                self.db = pickle.load(f)
        else:
            self.db = []
            print(f"⚠️ Database file not found: {db_path}")
        
        # โหลด Neural Network
        self.nn_identifier = None
        if config.USE_NEURAL_NETWORK:
            if nn_model_path and os.path.exists(nn_model_path):
                self.nn_identifier = NeuralDrugIdentifier(nn_model_path, device='cpu')
            else:
                print("⚠️ Neural model not found, using SIFT only")
        
        # โหลด SIFT
        self.sift_identifier = SIFTIdentifier()
        
        # จัดโครงสร้าง database
        self.nn_features_db = {}
        self.sift_db = {}
        
        # ตรวจสอบว่า database เป็น format ใหม่หรือเก่า
        if len(self.db) > 0:
            first_item = self.db[0]
            
            # Format ใหม่: มี nn_features
            if 'nn_features' in first_item:
                self._load_neural_database()
            # Format เก่า: มีแต่ SIFT
            else:
                self._load_legacy_database()
        
        print(f"✅ Database loaded: {len(self.db)} entries")
        print(f"   - Neural features: {len(self.nn_features_db)} drugs")
        print(f"   - SIFT features: {len(self.sift_db)} drugs")
    
    def _load_neural_database(self):
        """โหลด database format ใหม่ (มี Neural Network features)"""
        for item in self.db:
            name = item['name']
            
            # Neural Network features
            if 'nn_features' in item and item['nn_features'] is not None:
                if name not in self.nn_features_db:
                    self.nn_features_db[name] = []
                self.nn_features_db[name].append(np.array(item['nn_features']))
            
            # SIFT features (สำหรับ backward compatibility)
            if 'sift_des' in item and item['sift_des'] is not None:
                if name not in self.sift_db:
                    self.sift_db[name] = []
                self.sift_db[name].append(np.array(item['sift_des']))
    
    def _load_legacy_database(self):
        """โหลด database format เก่า (มีแต่ SIFT)"""
        for item in self.db:
            name = item['name']
            
            # SIFT features
            if 'des' in item and item['des'] is not None:
                if name not in self.sift_db:
                    self.sift_db[name] = []
                self.sift_db[name].append(np.array(item['des']))
    
    def search(self, identifier, query_img, target_drugs=None, sift_ratio_threshold=0.75):
        """Search ด้วย hybrid approach"""
        if config.USE_HYBRID_MATCHING and self.nn_identifier and self.nn_features_db:
            return self._search_hybrid(query_img, target_drugs)
        else:
            # Fallback to SIFT
            return self._search_sift(identifier, query_img, target_drugs, sift_ratio_threshold)
    
    def _search_hybrid(self, query_img, target_drugs=None):
        """Search ด้วยวิธี hybrid (Neural + SIFT)"""
        best_match = None
        best_score = -1
        
        # Extract neural features จาก query
        query_features = self.nn_identifier.extract_features(query_img)
        if query_features is None:
            return None
        
        # สร้าง partial crops สำหรับ query
        partial_crops = self.nn_identifier.create_partial_crops(query_img)
        
        drugs_to_search = target_drugs if target_drugs else list(self.nn_features_db.keys())
        
        for drug_name in drugs_to_search:
            if drug_name not in self.nn_features_db:
                continue
            
            drug_features_list = self.nn_features_db[drug_name]
            
            # คำนวณ similarity กับทุก features ของยาตัวนี้
            max_drug_score = -1
            
            for db_features in drug_features_list:
                # คำนวณ similarity
                similarity = self.nn_identifier.compute_similarity(query_features, db_features)
                
                # ถ้า similarity ต่ำ ให้ลองกับ partial crops
                if similarity < config.NEURAL_MIN_CONFIDENCE and len(partial_crops) > 1:
                    for partial_crop in partial_crops[1:]:  # ข้ามภาพแรกที่เป็นภาพเต็ม
                        if partial_crop.size == 0:
                            continue
                        
                        # Extract features จาก partial crop
                        partial_features = self.nn_identifier.extract_features(partial_crop)
                        if partial_features is not None:
                            partial_similarity = self.nn_identifier.compute_similarity(
                                partial_features, db_features
                            )
                            similarity = max(similarity, partial_similarity)
                
                if similarity > max_drug_score:
                    max_drug_score = similarity
            
            # ใช้ SIFT เป็นตัวเสริมถ้า similarity ต่ำ
            if max_drug_score < config.NEURAL_THRESHOLD and drug_name in self.sift_db:
                sift_score = self._compute_sift_score(query_img, drug_name)
                if sift_score > 0:
                    # Combine scores
                    combined_score = (max_drug_score * config.HYBRID_NN_WEIGHT + 
                                     sift_score * config.HYBRID_SIFT_WEIGHT)
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = drug_name
            
            # ถ้า neural score สูงพอ
            elif max_drug_score > best_score:
                best_score = max_drug_score
                best_match = drug_name
        
        if best_match and best_score >= config.NEURAL_MIN_CONFIDENCE:
            return {
                'name': best_match,
                'score': best_score,
                'method': 'hybrid'
            }
        
        return None
    
    def _compute_sift_score(self, query_img, drug_name):
        """คำนวณ SIFT score"""
        if drug_name not in self.sift_db:
            return 0
        
        query_kp, query_des = self.sift_identifier.extract_features(query_img)
        if query_des is None:
            return 0
        
        max_sift_score = 0
        
        for db_des in self.sift_db[drug_name]:
            if db_des is None or len(db_des) < 2:
                continue
            
            try:
                matches = self.sift_identifier.flann.knnMatch(query_des, db_des, k=2)
            except:
                continue
            
            good_matches = []
            for m, n in matches:
                if m.distance < config.SIFT_MATCH_RATIO * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) > max_sift_score:
                max_sift_score = len(good_matches)
        
        # Normalize SIFT score (0-1)
        normalized_score = min(max_sift_score / 50.0, 1.0)
        return normalized_score
    
    def _search_sift(self, identifier, query_img, target_drugs=None, sift_ratio_threshold=0.75):
        """Search ด้วย SIFT แบบเดิม (สำหรับ backward compatibility)"""
        # ใช้ logic เดิมจาก VectorDB
        query_kp, query_des = identifier.extract_features(query_img)
        
        if query_des is None:
            return None
        
        best_match = None
        best_inliers = 0
        
        drugs_to_search = target_drugs if target_drugs else list(self.sift_db.keys())
        
        for drug_name in drugs_to_search:
            if drug_name not in self.sift_db:
                continue
            
            for db_des in self.sift_db[drug_name]:
                if db_des is None:
                    continue
                
                try:
                    matches = identifier.flann.knnMatch(query_des, db_des, k=2)
                except:
                    continue
                
                good_matches = []
                for m, n in matches:
                    if m.distance < sift_ratio_threshold * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) >= 4:
                    src_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if mask is not None:
                        inliers = mask.ravel().tolist().count(1)
                        if inliers > best_inliers and inliers >= config.SIFT_MIN_MATCH_COUNT:
                            best_inliers = inliers
                            best_match = drug_name
        
        if best_match:
            return {
                'name': best_match,
                'score': best_inliers,
                'method': 'sift'
            }
        
        return None