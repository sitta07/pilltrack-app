import cv2
import time
import os
import numpy as np
import threading
import config
from engines import YOLODetector, SIFTIdentifier
from database import VectorDB
from his_mock import HISSystem

# ✅ Import Picamera2 (Library กล้อง Native ของ Pi 5)
try:
    from picamera2 import Picamera2
except ImportError:
    print("❌ Error: Picamera2 not found. Make sure you are on Raspberry Pi OS.")

# ==========================================
# 🧵 CLASS: WebcamStream (Picamera2 Engine)
# ==========================================
class WebcamStream:
    def __init__(self):
        print("📸 Initializing Picamera2...")
        self.picam2 = Picamera2()

        # 1. Config: ตั้งค่าให้ส่งภาพ RGB888 ขนาด 640x480 (เบาและเร็ว)
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameDurationLimits": (33333, 33333)} # Lock ~30 FPS
        )
        self.picam2.configure(config)
        self.picam2.start()

        # 2. Tuning: ปรับ Auto Focus/White Balance
        self.picam2.set_controls({
            "AwbMode": 0,       # 0 = Auto
            "AeMeteringMode": 0 # 0 = CentreWeighted
        })
        
        print("⏳ Camera warming up (2s)...")
        time.sleep(2.0)
        
        # ลองจับภาพแรกเพื่อเช็ค
        self.frame = self.convert_frame(self.picam2.capture_array())
        self.stopped = False

    def convert_frame(self, raw_frame):
        # Picamera2 ส่งมาเป็น RGB แต่ OpenCV ชอบ BGR -> ต้องกลับสี
        return cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            try:
                # ดึงภาพดิบ
                raw = self.picam2.capture_array()
                # แปลงสีและเก็บลงตัวแปรหลัก
                self.frame = self.convert_frame(raw)
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
# 🚀 MAIN LOOP
# ==========================================
def main():
    print("🚀 Starting PillTrack (Headless Mode - No GUI)...")
    
    # 1. Load Engines
    # พยายามหา ONNX ก่อน
    model_path = config.MODEL_YOLO_PATH.replace('.pt', '.onnx')
    if not os.path.exists(model_path):
        print("⚠️ ONNX not found, using .pt")
        model_path = config.MODEL_YOLO_PATH
        
    yolo = YOLODetector(model_path)
    identifier = SIFTIdentifier()
    db = VectorDB()
    his = HISSystem()
    
    # 2. Setup Patient Data
    current_patient_id = "HN001" 
    target_drug_list = his.get_patient_drugs(current_patient_id)

    # 3. Start Camera
    try:
        vs = WebcamStream().start()
        print("✅ Camera Started! Processing...")
    except Exception as e:
        print(f"❌ Camera Failed: {e}")
        return

    fps_avg = 0
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # รับภาพ
            frame = vs.read()
            if frame is None: continue

            loop_start = time.time()
            img_area = frame.shape[0] * frame.shape[1]
            found_drugs = [] # เก็บรายชื่อยาที่เจอในเฟรมนี้

            # --- A. DETECT ---
            # ใช้การตั้งค่าแบบเข้มงวดเพื่อลดขยะ
            results = yolo.detect(frame, conf=0.60, iou=0.20, agnostic_nms=True, max_det=15)
            
            for i, box in enumerate(results.boxes):
                # --- B. FILTER NOISE ---
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # กรองขนาด: เล็กกว่า 2% ของภาพ -> ข้าม
                box_area = (x2-x1) * (y2-y1)
                if box_area < (img_area * 0.02): continue 

                # กรองสัดส่วน (Aspect Ratio)
                w, h = (x2-x1), (y2-y1)
                if h == 0: continue
                aspect = w / h
                if aspect > 5.0 or aspect < 0.2: continue

                # --- C. CROP & SEARCH ---
                mask = results.masks[i] if results.masks else None
                crop_img = yolo.get_crop(frame, box, mask)
                
                match_result = db.search(identifier, crop_img, target_drugs=target_drug_list)
                
                if match_result:
                    # ถ้าเจอ ให้เก็บชื่อยาไว้โชว์
                    found_drugs.append(f"{match_result['name']} ({match_result['inliers']})")

            # --- D. REPORT STATUS (NO GUI) ---
            # คำนวณ FPS
            fps_avg = 1.0 / (time.time() - loop_start)
            
            # สร้างข้อความ Status
            status_msg = "Searching..."
            if found_drugs:
                status_msg = f"🟢 FOUND: {', '.join(found_drugs)}"
            
            # ปริ้นบรรทัดเดียว (ใช้ \r เพื่อเขียนทับบรรทัดเดิม ไม่ให้รก Terminal)
            print(f"\rFPS: {fps_avg:.1f} | {status_msg}" + " " * 20, end="", flush=True)

            # ❌ ปิดการแสดงผลภาพเพื่อป้องกัน SSH หลุด
            # cv2.imshow("PillTrack Pi 5", frame)
            # if cv2.waitKey(1) == ord('q'): break
            
            # ใช้ Ctrl+C เพื่อหยุดโปรแกรมแทน

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")

    finally:
        vs.stop()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()