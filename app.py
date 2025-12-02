import cv2
import time
import os
import numpy as np
import threading
import config
from engines import YOLODetector, SIFTIdentifier
from database import VectorDB
from his_mock import HISSystem

# ✅ POWER SAVING TRICK 1: บังคับใช้ CPU แค่ 1 Core
# เพื่อไม่ให้กินไฟกระชาก 4 Core พร้อมกัน
try:
    import torch
    torch.set_num_threads(1) 
    print("🔋 Power Saving: Restricted PyTorch to 1 CPU Core")
except ImportError:
    pass

try:
    from picamera2 import Picamera2
except ImportError:
    print("❌ Error: Picamera2 not found. Run on Raspberry Pi OS.")

# ==========================================
# 🧵 CLASS: WebcamStream (Low Res Mode)
# ==========================================
class WebcamStream:
    def __init__(self):
        print("📸 Initializing Picamera2 (Low Power Mode)...")
        self.picam2 = Picamera2()

        # ✅ POWER SAVING TRICK 2: ลดความละเอียดกล้องเหลือ 320x240
        # เล็กหน่อย แต่ประหยัด Bandwidth และ Ram
        config = self.picam2.create_preview_configuration(
            main={"size": (320, 240), "format": "RGB888"},
            controls={"FrameDurationLimits": (66666, 66666)} # Lock ~15 FPS (ไม่เอา 30)
        )
        self.picam2.configure(config)
        self.picam2.start()

        self.picam2.set_controls({
            "AwbMode": 0,
            "AeMeteringMode": 0
        })
        
        time.sleep(2.0)
        self.frame = self.convert_frame(self.picam2.capture_array())
        self.stopped = False

    def convert_frame(self, raw_frame):
        return cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            try:
                raw = self.picam2.capture_array()
                self.frame = self.convert_frame(raw)
                # ✅ POWER SAVING TRICK 3: ให้ Thread กล้องพักบ้าง
                time.sleep(0.05) 
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
    print("🚀 Starting PillTrack (Survival Mode - Low Voltage Safe)...")
    
    # Check Model
# 1. Load Engines
    # ✅ FORCE .PT MODE: บังคับใช้ไฟล์ .pt ตรงๆ เพื่อตัดปัญหา ONNX Error
    print("forcing load .pt model...")
    # ตรวจสอบว่าชื่อไฟล์ใน config.py ถูกต้อง (ต้องลงท้ายด้วย .pt)
    if config.MODEL_YOLO_PATH.endswith('.onnx'):
        # ถ้าเผลอตั้งเป็น .onnx ให้แก้กลับเป็น .pt เอง
        model_path = config.MODEL_YOLO_PATH.replace('.onnx', '.pt')
    else:
        model_path = config.MODEL_YOLO_PATH

    print(f"👉 Loading Model: {model_path}")
    yolo = YOLODetector(model_path)
    identifier = SIFTIdentifier()
    db = VectorDB()
    his = HISSystem()
    
    target_drug_list = his.get_patient_drugs("HN001") 

    try:
        vs = WebcamStream().start()
        print("✅ Camera Started!")
    except Exception as e:
        print(f"❌ Camera Failed: {e}")
        return

    fps_avg = 0
    
    try:
        while True:
            frame = vs.read()
            if frame is None: continue

            loop_start = time.time()
            img_area = frame.shape[0] * frame.shape[1]
            found_drugs = []

            # --- A. DETECT ---
            # ✅ POWER SAVING TRICK 4: ลด imgsz เหลือ 320
            # ภาพเล็กลงครึ่งนึง กินไฟน้อยลงเยอะ
            results = yolo.detect(frame, 
                                  conf=0.60, 
                                  iou=0.20, 
                                  agnostic_nms=True, 
                                  max_det=10,
                                  imgsz=320) # <--- สำคัญ!
            
            for i, box in enumerate(results.boxes):
                # --- B. FILTER ---
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                box_area = (x2-x1) * (y2-y1)
                if box_area < (img_area * 0.05): continue # กรองขยะเล็กๆ ทิ้งไวๆ (5%)

                w, h = (x2-x1), (y2-y1)
                if h == 0: continue
                aspect = w / h
                if aspect > 5.0 or aspect < 0.2: continue

                # --- C. CROP & SEARCH ---
                mask = results.masks[i] if results.masks else None
                crop_img = yolo.get_crop(frame, box, mask)
                
                match_result = db.search(identifier, crop_img, target_drugs=target_drug_list)
                
                if match_result:
                    found_drugs.append(f"{match_result['name']}")

            # --- D. REPORT ---
            fps_avg = 1.0 / (time.time() - loop_start)
            status_msg = "Scanning..."
            if found_drugs:
                status_msg = f"🟢 FOUND: {', '.join(found_drugs)}"
            
            print(f"\rFPS: {fps_avg:.1f} | {status_msg}" + " " * 20, end="", flush=True)
            
            # ✅ POWER SAVING TRICK 5: บังคับให้ CPU พักหายใจ 0.1 วินาทีทุกรอบ
            # อันนี้ช่วยลดความร้อนและไฟกระชากได้ดีที่สุด
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")

    finally:
        vs.stop()

if __name__ == "__main__":
    main()