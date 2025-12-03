import cv2
import numpy as np
from ultralytics import YOLO
import config

class YOLODetector:
    def __init__(self, model_path, task=None):
        print(f"🔄 Loading YOLO Model: {model_path}...")
        self.model = YOLO(model_path, task=task)
    
    # รองรับ kwargs เพื่อให้รับค่า imgsz หรือ conf จาก app ได้ยืดหยุ่น
    def detect(self, frame, conf=0.1, iou=0.25, agnostic_nms=True, max_det=100, **kwargs):
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
        
        # ใช้ CLAHE ช่วยเรื่องแสงสะท้อนบนแผงยา (สูตรเมื่อวาน)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def extract_features(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0: return [], None
        
        # 1. แปลงเป็น Grayscale
        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr
            
        # 2. ปรับแสง (CLAHE)
        gray = self.clahe.apply(gray)
        
        # 3. Detect SIFT
        kp, des = self.sift.detectAndCompute(gray, None)
        
        # ✅ คืนค่าแค่ 2 ตัว (ไม่มีสี)
        return kp, des

    def compare(self, query_pack, db_pack):
        kp_q, des_q = query_pack
        kp_db, des_db = db_pack
        
        if des_q is None or des_db is None or len(des_q) < 2 or len(des_db) < 2:
            return 0
            
        try:
            matches = self.flann.knnMatch(des_q, des_db, k=2)
        except: return 0

        good_matches = []
        for m, n in matches:
            if m.distance < config.SIFT_MATCH_RATIO * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 4: return 0
            
        src_pts = np.float32([kp_q[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_db[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None: return 0
        
        return mask.ravel().tolist().count(1)