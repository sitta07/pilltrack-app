import cv2
import time
import os
import numpy as np
import threading
import sys

# ✅ FIX Display
os.environ["QT_QPA_PLATFORM"] = "xcb"

try:
    import config
    from engines import YOLODetector, SIFTIdentifier
    from database import VectorDB
    from his_mock import HISSystem
    from picamera2 import Picamera2
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# ==========================================
# 📷 WEBCAM STREAM (30 FPS Limited)
# ==========================================
class WebcamStream:
    def __init__(self):
        self.stopped = False
        self.frame = None
        self.picam2 = None

    def start(self):
        print("📷 Camera: HD Mode @ 30 FPS")
        try:
            self.picam2 = Picamera2()
            # ✅ Force 30 FPS (33333 microseconds)
            config = self.picam2.create_preview_configuration(
                main={"size": (1280, 720), "format": "RGB888"},
                controls={"FrameDurationLimits": (33333, 33333)} 
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(1.0) # Warmup เร็วขึ้น
        except Exception as e:
            print(f"❌ Camera Error: {e}")
            self.stopped = True
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                # capture_array รอเฟรมใหม่มา (Block จนกว่าจะครบ 33ms)
                frame = self.picam2.capture_array()
                if frame is not None: self.frame = frame
            except: pass

    def read(self): return self.frame
    def stop(self): self.stopped = True; self.picam2.stop()

# ==========================================
# 🧠 ASYNC AI WORKER
# ==========================================
class AsyncDetector:
    def __init__(self, model_path, patient_drugs):
        # ✅ ใช้ engines.py ตัวใหม่ (YOLODetector ที่แก้แล้ว)
        self.yolo = YOLODetector(model_path)
        self.identifier = SIFTIdentifier()
        self.db = VectorDB()
        self.patient_drugs = patient_drugs
        
        self.latest_frame = None
        self.verified_drugs = set()
        self.running = True
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()
        return self

    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()

    def get_results(self):
        return self.verified_drugs

    def run(self):
        print("🧠 AI Worker Running (High Speed)...")
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None

            if frame_to_process is not None:
                h, w = frame_to_process.shape[:2]
                
                # 1. Detect (ใช้ Fast YOLO จาก engines.py)
                # คืนค่าเป็น List ของ [x1, y1, x2, y2]
                boxes = self.yolo.detect(frame_to_process, conf=0.6)
                
                # 2. Filter & Sort (เอาเม็ดใหญ่สุด 1 เม็ด)
                valid_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    area = (x2-x1)*(y2-y1)
                    if area > (w*h * 0.01): 
                         valid_boxes.append((area, box))
                
                valid_boxes.sort(key=lambda x: x[0], reverse=True)
                target_boxes = valid_boxes[:1]

                # 3. SIFT Compare
                current_found = set()
                for _, box in target_boxes:
                    # Crop เร็ว (ไม่มี Mask)
                    crop_img = self.yolo.get_crop(frame_to_process, box)
                    match_result = self.db.search(self.identifier, crop_img, target_drugs=self.patient_drugs)
                    if match_result:
                        current_found.add(match_result['name'])
                
                if current_found:
                    self.verified_drugs.update(current_found)
            else:
                # Sleep นานขึ้นนิดนึง เพื่อคืน CPU ให้ระบบ
                time.sleep(0.02)

    def stop(self): self.running = False

# ==========================================
# 🎨 UI (Clean - No Drawing)
# ==========================================
def draw_ui(img, patient_info, found_set, fps):
    h, w = img.shape[:2]
    
    # ❌ ไม่วาด Box หรือ Dot แล้ว (ตามสั่ง)

    # Info (FPS + Temp)
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read())/1000.0
    except: temp = 0.0
    
    cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(img, f"TEMP: {temp:.1f}C", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if temp<80 else (0,0,255), 2)

    # Panel
    panel_w = 280
    x1, y1 = w - panel_w - 20, 20
    x2, y2 = w - 20, y1 + 100 + (len(patient_info['drugs']) * 35)
    
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    
    cv2.putText(img, "PATIENT INFO", (x1+10, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(img, f"{patient_info['name']}", (x1+10, y1+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    sy = y1 + 100
    for drug in patient_info['drugs']:
        found = any(drug.lower() in f.lower() for f in found_set)
        icon, color = ("[/]", (0,255,0)) if found else ("[ ]", (150,150,150))
        cv2.putText(img, f"{icon} {drug}", (x1+10, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        sy += 30

# ==========================================
# 🚀 MAIN
# ==========================================
def main():
    print("🚀 Starting PillTrack (Max Speed Mode)...")
    
    # Auto ONNX Path Fix
    if config.MODEL_YOLO_PATH.endswith('.pt'):
        model_path = config.MODEL_YOLO_PATH.replace('.pt', '.onnx')
    else:
        model_path = config.MODEL_YOLO_PATH
        
    if not os.path.exists(model_path):
        print(f"❌ ONNX file not found: {model_path}")
        sys.exit(1)

    his = HISSystem()
    p_data = his.get_patient_info("HN001")
    p_info = {"hn": "HN001", "name": p_data['name'], "drugs": p_data['drugs']}

    ai_worker = AsyncDetector(model_path, p_info['drugs']).start()
    vs = WebcamStream().start()
    
    print("⏳ Waiting for camera...")
    while vs.read() is None: time.sleep(0.1)
    
    cv2.namedWindow("PillTrack", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("PillTrack", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_time = 0
    try:
        while True:
            frame = vs.read()
            if frame is None: continue
            
            # ส่งภาพไปให้ AI ทำงานเบื้องหลัง
            ai_worker.update_frame(frame)
            found = ai_worker.get_results()
            
            # คำนวณ FPS
            curr = time.time()
            fps = 1/(curr-prev_time) if curr>prev_time else 0
            prev_time = curr
            
            # วาดหน้าจอ (แค่ Panel)
            ui = frame.copy()
            draw_ui(ui, p_info, found, fps)
            
            cv2.imshow("PillTrack", ui)
            if cv2.waitKey(1) == ord('q'): break
            
            # Sleep เล็กน้อยเพื่อให้ CPU ไม่ 100% ตลอดเวลา
            time.sleep(0.01)

    except KeyboardInterrupt: pass
    finally:
        ai_worker.stop()
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()