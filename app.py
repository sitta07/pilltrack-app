import cv2
import time
import os
import numpy as np
import threading
import config
from engines import YOLODetector, SIFTIdentifier
from database import VectorDB
from his_mock import HISSystem

# ==========================================
# 🧵 CLASS: WebcamStream (Tuned for RPi 5)
# ==========================================
class WebcamStream:
    def __init__(self, src=0):
        # ✅ PI OPTIMIZATION 1: ใช้ backend V4L2
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        
        # ✅ PI OPTIMIZATION 2: ลดความละเอียด input เพื่อ FPS สูงสุด
        # YOLO รับภาพ 640x640 การส่งภาพ 4K ไปให้มันย่อเสียเวลาเปล่าครับ
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # เช็คว่าเปิดกล้องติดไหม
        if not self.stream.isOpened():
            print("❌ Error: Could not open camera. Check connection!")
            # ลอง fallback ไปใช้ค่า default เผื่อ V4L2 มีปัญหา
            self.stream.open(src)
            
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

# ==========================================
# 🎨 DASHBOARD
# ==========================================
def draw_dashboard(img, match_result, fps):
    # วาด FPS ตัวใหญ่ๆ สีเหลือง
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 255), 2)
    return img

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
def main():
    print("🚀 Starting PillTrack on Raspberry Pi 5...")
    
    # 1. Load Engines
    # พยายามโหลด ONNX ก่อน
    model_path = config.MODEL_YOLO_PATH.replace('.pt', '.onnx')
    if not os.path.exists(model_path):
        print("⚠️ ONNX not found, using .pt (Slower on Pi)")
        model_path = config.MODEL_YOLO_PATH
        
    yolo = YOLODetector(model_path)
    identifier = SIFTIdentifier()
    db = VectorDB()
    his = HISSystem()
    
    # 2. Setup Data
    current_patient_id = "HN001" 
    target_drug_list = his.get_patient_drugs(current_patient_id)
    # target_drug_list = None # ปลดล็อคถ้าจะหาทุกอย่าง

    # 3. Start Camera
    if config.USE_CAMERA:
        print("📷 Camera Starting (Warmup 2s)...")
        vs = WebcamStream(src=config.CAMERA_ID).start()
        time.sleep(2.0)
    else:
        print("❌ Error: Config is not set to use Camera")
        return

    fps_avg = 0
    
    while True:
        # รับภาพจาก Thread
        frame = vs.read()
        if frame is None: 
            print("⚠️ Frame not received")
            continue

        loop_start = time.time()
        annotated_frame = frame.copy()
        img_area = frame.shape[0] * frame.shape[1]

        # --- A. DETECT ---
        # บน Pi อาจจะต้องลด max_det ลงอีกเพื่อประหยัด CPU
        results = yolo.detect(frame, conf=0.60, iou=0.20, agnostic_nms=True, max_det=15)
        
        for i, box in enumerate(results.boxes):
            # --- B. FILTER NOISE ---
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # กรองขนาด: เล็กกว่า 2% ของภาพ -> ทิ้ง
            box_area = (x2-x1) * (y2-y1)
            if box_area < (img_area * 0.02): continue 

            # กรองสัดส่วน
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
                # กรอบเขียว
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # ชื่อยา
                label = f"{match_result['name']} ({match_result['inliers']})"
                # ปรับตำแหน่งตัวหนังสือให้ไม่ตกขอบ
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.putText(annotated_frame, label, (x1, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Segmentation Mask
                if mask is not None:
                    mask_raw = mask.data[0].cpu().numpy()
                    mask_rs = cv2.resize(mask_raw, (frame.shape[1], frame.shape[0]))
                    mask_bin = (mask_rs > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated_frame, contours, -1, (0, 255, 255), 1)

        fps_avg = 1.0 / (time.time() - loop_start)
        annotated_frame = draw_dashboard(annotated_frame, None, fps_avg)
        
        # Show Result
        cv2.imshow("PillTrack Pi 5", annotated_frame)
        
        if cv2.waitKey(1) == ord('q'): break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()