import cv2
import time
import os
import numpy as np
import threading
import config
from engines import YOLODetector, SIFTIdentifier
from database import VectorDB
from his_mock import HISSystem

# ✅ Import Library กล้องของ Pi 5
try:
    from picamera2 import Picamera2
except ImportError:
    print("❌ Error: Picamera2 not found. Please run on Raspberry Pi OS.")

# ==========================================
# 🧵 CLASS: WebcamStream (Picamera2 Native)
# ==========================================
class WebcamStream:
    def __init__(self):
        print("📸 Initializing Picamera2...")
        self.picam2 = Picamera2()

        # Config ให้เป็น BGR888 (เพื่อให้ OpenCV เอาไปใช้ได้เลย ไม่ต้องแปลงสี)
        # Size 640x480 เพื่อความเร็ว
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "BGR888"},
            controls={"FrameDurationLimits": (33333, 33333)} # Lock ~30 FPS
        )
        self.picam2.configure(config)
        self.picam2.start()

        # Tuning (Auto Focus/White Balance)
        self.picam2.set_controls({
            "AwbMode": 0,       # Auto White Balance
            "AeMeteringMode": 0 # Center Weighted
        })
        
        # รอ Warmup แป๊บนึง
        time.sleep(1.0)
        
        self.frame = self.picam2.capture_array()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            # ดึงภาพแบบ Array (เร็วมาก)
            try:
                self.frame = self.picam2.capture_array()
            except Exception as e:
                print(f"Camera Error: {e}")
                self.stopped = True

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.picam2.stop()
        self.picam2.close()

# ==========================================
# 🎨 DASHBOARD
# ==========================================
def draw_dashboard(img, match_result, fps):
    # วาด FPS
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 255), 2)
    return img

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
def main():
    print("🚀 Starting PillTrack on Raspberry Pi 5 (Picamera2 Engine)...")
    
    # 1. Load Engines
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
    # target_drug_list = None 

    # 3. Start Camera (Picamera2)
    if config.USE_CAMERA:
        try:
            # ไม่ต้องใส่ src เพราะ Picamera2 หาเอง
            vs = WebcamStream().start()
            print("✅ Camera Started!")
        except Exception as e:
            print(f"❌ Camera Failed: {e}")
            return
    else:
        print("❌ Config USE_CAMERA is False")
        return

    fps_avg = 0
    
    while True:
        # รับภาพ
        frame = vs.read()
        if frame is None: continue

        loop_start = time.time()
        annotated_frame = frame.copy()
        img_area = frame.shape[0] * frame.shape[1]

        # --- A. DETECT ---
        # iou=0.20 เพื่อลดกรอบซ้อน
        results = yolo.detect(frame, conf=0.60, iou=0.20, agnostic_nms=True, max_det=15)
        
        for i, box in enumerate(results.boxes):
            # --- B. FILTER NOISE ---
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            box_area = (x2-x1) * (y2-y1)
            if box_area < (img_area * 0.02): continue 

            w, h = (x2-x1), (y2-y1)
            if h == 0: continue
            aspect = w / h
            if aspect > 5.0 or aspect < 0.2: continue

            # --- C. CROP & SEARCH ---
            mask = results.masks[i] if results.masks else None
            crop_img = yolo.get_crop(frame, box, mask)
            
            match_result = db.search(identifier, crop_img, target_drugs=target_drug_list)
            
            # --- D. VISUALIZE (GREEN ONLY) ---
            if match_result:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"{match_result['name']} ({match_result['inliers']})"
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.putText(annotated_frame, label, (x1, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if mask is not None:
                    mask_raw = mask.data[0].cpu().numpy()
                    mask_rs = cv2.resize(mask_raw, (frame.shape[1], frame.shape[0]))
                    mask_bin = (mask_rs > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated_frame, contours, -1, (0, 255, 255), 1)

        fps_avg = 1.0 / (time.time() - loop_start)
        annotated_frame = draw_dashboard(annotated_frame, None, fps_avg)
        
        cv2.imshow("PillTrack Pi 5", annotated_frame)
        
        if cv2.waitKey(1) == ord('q'): break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()