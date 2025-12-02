import cv2
import time
import os
import numpy as np
import threading
import config
from engines import YOLODetector, SIFTIdentifier
from database import VectorDB
from his_mock import HISSystem

# ✅ POWER SAVING 1: บังคับใช้ CPU 1 Core
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

        # ✅ POWER SAVING 2: ลดความละเอียดเหลือ 320x240
        config = self.picam2.create_preview_configuration(
            main={"size": (320, 240), "format": "RGB888"},
            controls={"FrameDurationLimits": (66666, 66666)} # Lock ~15 FPS
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
                # ✅ POWER SAVING 3: พัก Thread กล้องนิดนึง
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
# 🎨 DASHBOARD
# ==========================================
def draw_dashboard(img, fps):
    # วาด FPS มุมซ้ายบน
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 255), 2)
    return img

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
def main():
    print("🚀 Starting PillTrack (Survival Mode + GUI)...")
    
    # ✅ FORCE .PT MODE: บังคับใช้ไฟล์ .pt
    print("👉 Forcing load .pt model...")
    if config.MODEL_YOLO_PATH.endswith('.onnx'):
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
            annotated_frame = frame.copy()
            img_area = frame.shape[0] * frame.shape[1]
            found_drugs = []

            # --- A. DETECT ---
            # ✅ POWER SAVING 4: imgsz=320, max_det=10
            results = yolo.detect(frame, 
                                  conf=0.60, 
                                  iou=0.20, 
                                  agnostic_nms=True, 
                                  max_det=10,
                                  imgsz=320)
            
            for i, box in enumerate(results.boxes):
                # --- B. FILTER NOISE ---
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                box_area = (x2-x1) * (y2-y1)
                if box_area < (img_area * 0.05): continue 

                w, h = (x2-x1), (y2-y1)
                if h == 0: continue
                aspect = w / h
                if aspect > 5.0 or aspect < 0.2: continue

                # --- C. CROP & SEARCH ---
                mask = results.masks[i] if results.masks else None
                crop_img = yolo.get_crop(frame, box, mask)
                
                match_result = db.search(identifier, crop_img, target_drugs=target_drug_list)
                
                # --- D. VISUALIZE (วาดกรอบเขียว) ---
                if match_result:
                    found_drugs.append(match_result['name'])
                    
                    # วาดกรอบ
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # วาดชื่อยา
                    label = f"{match_result['name']} ({match_result['inliers']})"
                    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 15
                    cv2.putText(annotated_frame, label, (x1, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- E. SHOW RESULT ---
            fps_avg = 1.0 / (time.time() - loop_start)
            annotated_frame = draw_dashboard(annotated_frame, fps_avg)
            
            # ปริ้น Status ลง Terminal ด้วย
            status_msg = f"🟢 FOUND: {', '.join(found_drugs)}" if found_drugs else "Scanning..."
            print(f"\rFPS: {fps_avg:.1f} | {status_msg}" + " " * 10, end="", flush=True)
            
            # ✅ เปิดหน้าต่างภาพ (GUI)
            cv2.imshow("PillTrack Pi 5 (Survival)", annotated_frame)
            if cv2.waitKey(1) == ord('q'): break
            
            # ✅ POWER SAVING 5: พักหายใจ 0.1 วิ
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")

    finally:
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()