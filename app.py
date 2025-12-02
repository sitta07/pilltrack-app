import cv2
import time
import os
import numpy as np
import threading
import config
from engines import YOLODetector, SIFTIdentifier
from database import VectorDB
from his_mock import HISSystem

# ✅ Import ตรงๆ เลย (ถ้า Environment ถูก ต้องไม่ Error)
from picamera2 import Picamera2

# ==========================================
# 🧵 CLASS: WebcamStream (Based on cam_pro_7.py)
# ==========================================
class WebcamStream:
    def __init__(self):
        print("📸 Initializing Picamera2 (cam_pro_7 logic)...")
        self.picam2 = Picamera2()

        # 1. Config: อ้างอิงตาม cam_pro_7.py เป๊ะๆ
        # ใช้ RGB888 ตามที่คุณเทสต์ผ่าน
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameDurationLimits": (33333, 33333)} # Lock ~30 FPS
        )
        self.picam2.configure(config)
        self.picam2.start()

        # 2. Tuning: Set Controls ตาม cam_pro_7.py
        self.picam2.set_controls({
            "AwbMode": 0,       # 0 = Auto
            "AeMeteringMode": 0 # 0 = CentreWeighted
        })
        
        # Warm up 2 วินาที
        time.sleep(2.0)
        
        # ลองจับภาพแรก
        self.frame = self.picam2.capture_array()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            try:
                # 3. Capture: ดึงภาพแบบ Array
                raw_frame = self.picam2.capture_array()
                
                # ⚠️ สำคัญ: Picamera ส่งมาเป็น RGB แต่ OpenCV/YOLO ชอบ BGR
                # เราต้องแปลงสีนิดนึง ไม่งั้นสีเพี้ยน (ยาแดงจะเป็นน้ำเงิน)
                self.frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                
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
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 255), 2)
    return img

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
def main():
    print("🚀 Starting PillTrack on Raspberry Pi 5...")
    
    # 1. Load Engines
    model_path = config.MODEL_YOLO_PATH.replace('.pt', '.onnx')
    if not os.path.exists(model_path):
        print("⚠️ ONNX not found, using .pt")
        model_path = config.MODEL_YOLO_PATH
        
    yolo = YOLODetector(model_path)
    identifier = SIFTIdentifier()
    db = VectorDB()
    his = HISSystem()
    
    current_patient_id = "HN001" 
    target_drug_list = his.get_patient_drugs(current_patient_id)

    # 2. Start Camera
    try:
        vs = WebcamStream().start()
        print("✅ Camera Started!")
    except Exception as e:
        print(f"❌ Camera Failed to Start: {e}")
        return

    fps_avg = 0
    
    while True:
        frame = vs.read()
        if frame is None: continue

        loop_start = time.time()
        annotated_frame = frame.copy()
        img_area = frame.shape[0] * frame.shape[1]

        # --- A. DETECT ---
        results = yolo.detect(frame, conf=0.60, iou=0.20, agnostic_nms=True, max_det=15)
        
        for i, box in enumerate(results.boxes):
            # --- B. FILTER ---
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
            
            # --- D. VISUALIZE ---
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