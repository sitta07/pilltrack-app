import cv2
import time
import os
import numpy as np
import threading
import config
from engines import YOLODetector, SIFTIdentifier
from database import VectorDB
from his_mock import HISSystem

# --- Threaded Webcam ---
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- Dashboard (Green Only) ---
def draw_dashboard(img, match_result, fps):
    # FPS มุมซ้ายบน
    cv2.putText(img, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (0, 255, 255), 2)
    return img

# --- MAIN ---
def main():
    print("🚀 Starting PillTrack Phase 2 (Optimized + Clean Visuals)...")
    
    # 1. Load Engines
    # เช็ค ONNX ก่อน ถ้าไม่มีใช้ .pt
    model_path = config.MODEL_YOLO_PATH.replace('.pt', '.onnx')
    if not os.path.exists(model_path):
        print("⚠️ ONNX not found, using .pt")
        model_path = config.MODEL_YOLO_PATH
        
    yolo = YOLODetector(model_path)
    identifier = SIFTIdentifier()
    db = VectorDB()
    his = HISSystem()
    
    # 2. Setup Data
    current_patient_id = "HN001" 
    target_drug_list = his.get_patient_drugs(current_patient_id)
    # target_drug_list = None # ปลดคอมเมนต์ถ้าอยากหาทุกอย่าง

    # 3. Setup Camera/Images
    if config.USE_CAMERA:
        print("📷 Camera Starting...")
        vs = WebcamStream(src=config.CAMERA_ID).start()
        time.sleep(2.0)
    else:
        img_folder = 'test_match_real'
        if not os.path.exists(img_folder): os.makedirs(img_folder)
        test_images = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.png'))]
        test_images.sort()

    fps_avg = 0
    
    while True:
        if config.USE_CAMERA:
            frame = vs.read()
            if frame is None: break
        else:
            if not test_images: break
            path = os.path.join(img_folder, test_images.pop(0))
            frame = cv2.imread(path)
            if frame is None: continue
            print(f"\nProcessing: {path}")

        # Resize for speed
        if frame.shape[1] > 1280:
            scale = 1280 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
        loop_start = time.time()
        annotated_frame = frame.copy()
        img_area = frame.shape[0] * frame.shape[1]

        # --- A. DETECT (High Sensitivity Settings) ---
        # conf=0.60, iou=0.20 (ซ้อนกันนิดเดียวลบเลย), max_det=20
        results = yolo.detect(frame, conf=0.60, iou=0.20, agnostic_nms=True, max_det=20)
        
        for i, box in enumerate(results.boxes):
            # --- B. FILTER NOISE ---
            # 1. กรองขนาด: เล็กกว่า 2% ของภาพ -> ทิ้ง
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            box_area = (x2-x1) * (y2-y1)
            if box_area < (img_area * 0.02): continue 

            # 2. กรองสัดส่วน
            w, h = (x2-x1), (y2-y1)
            aspect = w / h
            if aspect > 5.0 or aspect < 0.2: continue

            # --- C. CROP & SEARCH ---
            mask = results.masks[i] if results.masks else None
            crop_img = yolo.get_crop(frame, box, mask)
            
            match_result = db.search(identifier, crop_img, target_drugs=target_drug_list)
            
            # --- D. VISUALIZE (GREEN ONLY) ---
            # วาดเฉพาะเมื่อเจอเท่านั้น!
            if match_result:
                # กรอบเขียว
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # ชื่อยา
                label = f"{match_result['name']} ({match_result['inliers']})"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Segmentation Mask เหลือง
                if mask is not None:
                    mask_raw = mask.data[0].cpu().numpy()
                    mask_rs = cv2.resize(mask_raw, (frame.shape[1], frame.shape[0]))
                    mask_bin = (mask_rs > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated_frame, contours, -1, (0, 255, 255), 2)

        fps_avg = 1.0 / (time.time() - loop_start)
        annotated_frame = draw_dashboard(annotated_frame, None, fps_avg)
        
        cv2.imshow("PillTrack Phase 2", annotated_frame)
        
        # Auto Slide 1.5s (สำหรับเทสต์รูป)
        delay = 1 if config.USE_CAMERA else 1500
        if cv2.waitKey(delay) == ord('q'): break

    if config.USE_CAMERA: vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()