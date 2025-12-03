import cv2
import numpy as np

# ==========================================
# ⚡ FAST YOLO (OpenCV DNN - No Masks)
# ==========================================
class YOLODetector:
    def __init__(self, model_path, task=None):
        # บังคับใช้ ONNX + OpenCV DNN เพื่อความเร็วสูงสุดบน Pi
        if not model_path.endswith('.onnx'):
             print(f"⚠️ Warning: {model_path} is not ONNX. It will be slow!")
        
        print(f"⚡ Loading Optimized YOLO: {model_path}...")
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # ตั้งค่า Input Size (320x320 เร็วสุด)
        self.input_size = (320, 320)

    def detect(self, frame, conf=0.5, iou=0.4, **kwargs):
        # 1. Prepare Input
        # swapRB=True เพราะ OpenCV DNN ชอบ BGR->RGB
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, self.input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # 2. Inference (เร็วมาก)
        out = self.net.forward()
        
        # 3. Post-Process (กรองข้อมูล)
        predictions = np.squeeze(out)
        # ถ้า Shape กลับด้าน ให้หมุนกลับ
        if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.transpose()
            
        # กรองด้วย Confidence (Vectorized - เร็ว)
        if predictions.shape[1] > 4:
            scores = np.max(predictions[:, 4:], axis=1)
            keep = scores >= conf
            predictions = predictions[keep]
            scores = scores[keep]
        else:
            return [] # ไม่เจออะไร

        if len(predictions) == 0:
            return []

        # แปลงเป็น Box [x, y, w, h] แบบ Pixel
        h_img, w_img = frame.shape[:2]
        scale_x = w_img / self.input_size[0]
        scale_y = h_img / self.input_size[1]
        
        boxes = []
        confidences = []
        
        for i, pred in enumerate(predictions):
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            left = int((cx - w/2) * scale_x)
            top = int((cy - h/2) * scale_y)
            width = int(w * scale_x)
            height = int(h * scale_y)
            boxes.append([left, top, width, height])
            confidences.append(float(scores[i]))

        # NMS (ลดกล่องซ้อน)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, iou)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                # ส่งกลับเป็น ultralytics format (Box object) เพื่อให้โค้ดเก่ารองรับได้ง่าย
                # แต่เราส่งเป็น list [x1, y1, x2, y2] ง่ายกว่า
                x, y, w, h = boxes[i]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_img, x+w), min(h_img, y+h)
                results.append((x1, y1, x2, y2))
                
        return results

    def get_crop(self, img, box, mask_data=None):
        # 🚀 FAST CROP: ตัดระบบ Mask ทิ้ง! (Mask กิน CPU เยอะมาก)
        # เราตัดสี่เหลี่ยมเลย เร็วกว่า 10 เท่า
        if isinstance(box, (list, tuple, np.ndarray)):
            x1, y1, x2, y2 = box
        else:
            # กรณีรับมาเป็น Object ของ ultralytics (เผื่อไว้)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
        return img[y1:y2, x1:x2]

# ==========================================
# ⚡ FAST SIFT (Reduced Features)
# ==========================================
class SIFTIdentifier:
    def __init__(self):
        print("⏳ Initializing Optimized SIFT...")
        # 🚀 ลด nfeatures จาก 2000 เหลือ 800 (เร็วขึ้น 2.5 เท่า)
        self.sift = cv2.SIFT_create(nfeatures=800) 
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=30) # ลด checks จาก 50 เหลือ 30
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def extract_features(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0: return [], None
        
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
        except: return 0

        good_matches = []
        for m, n in matches:
            # ลดความเข้มงวดลงนิดหน่อย (0.7 -> 0.75) เพื่อให้เจอข้อมูลง่ายขึ้น
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # 🚀 OPTIMIZATION: ถ้า Good Matches น้อยมาก ไม่ต้องทำ RANSAC (เสียเวลา)
        if len(good_matches) < 4: return 0
        
        # 🚀 OPTIMIZATION: ถ้า Good Matches เยอะพอแล้ว ตอบเลย! (ข้าม RANSAC)
        # RANSAC คือการคำนวณ Geometry ว่าจุดเรียงกันจริงไหม ซึ่งกิน CPU
        # ถ้าเราแค่ต้องการรู้ว่าเป็นยาตัวเดียวกัน จำนวนจุดที่ตรงกัน (Raw Count) ก็พอเชื่อถือได้แล้ว
        if len(good_matches) > 15: 
            return len(good_matches) # เร็วสุดๆ

        # ถ้าคะแนนก้ำกึ่ง ค่อยใช้ RANSAC ช่วยเช็ค
        src_pts = np.float32([kp_q[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_db[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None: return 0
        
        return mask.ravel().tolist().count(1)