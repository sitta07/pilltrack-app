import cv2
import time
import os
import numpy as np
import threading
import sys
import queue

# ✅ FIX Display
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Import Modules
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
# 📷 WEBCAM STREAM (60 FPS TUNED)
# ==========================================
class WebcamStream:
    def __init__(self):
        self.stopped = False
        self.frame = None
        self.grabbed = False
        self.picam2 = None

    def start(self):
        print("📷 Initializing Picamera2 (60 FPS Mode)...")
        try:
            self.picam2 = Picamera2()
            
            # ✅ Config: Force 60 FPS
            # FrameDurationLimits: (min_duration, max_duration) in microseconds
            # 1,000,000 / 60 = 16666 us
            config = self.picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                controls={"FrameDurationLimits": (16666, 16666)} 
            )
            self.picam2.configure(config)
            self.picam2.start()
            
            time.sleep(2.0)
            print("✅ Camera Running @ 60 FPS!")
            
        except Exception as e:
            print(f"❌ Camera Init Failed: {e}")
            self.stopped = True
            
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                # Capture แบบต่อเนื่องให้เร็วที่สุด
                frame = self.picam2.capture_array()
                if frame is not None:
                    self.frame = frame
                    self.grabbed = True
                else:
                    self.stopped = True
            except:
                self.stopped = True
            # ไม่ต้อง Sleep ตรงนี้เพื่อให้ได้ FPS สูงสุด

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except:
                pass

# ==========================================
# 🧠 ASYNC AI WORKER (แยก AI ไปรันอีกเลน)
# ==========================================
class AsyncDetector:
    def __init__(self, model_path, patient_drugs):
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
        # อัปเดตภาพล่าสุดที่จะให้ AI ตรวจ (Overrite ของเก่าได้เลยถ้า AI ทำไม่ทัน)
        with self.lock:
            self.latest_frame = frame.copy()

    def get_verified_drugs(self):
        return self.verified_drugs

    def run(self):
        print("🧠 AI Worker Started (Background)...")
        while self.running:
            frame_to_process = None
            
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None # Clear buffer

            if frame_to_process is not None:
                # 1. YOLO Detect (No Draw)
                results = self.yolo.detect(frame_to_process, conf=0.4, iou=0.45, agnostic_nms=True, max_det=5, imgsz=320)
                
                # 2. SIFT Check
                h, w = frame_to_process.shape[:2]
                current_found = set()
                
                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    if (x2-x1)*(y2-y1) < (w*h * 0.01): continue
                    
                    mask = results.masks[i] if results.masks else None
                    crop_img = self.yolo.get_crop(frame_to_process, box, mask)
                    
                    match_result = self.db.search(self.identifier, crop_img, target_drugs=self.patient_drugs)
                    if match_result:
                        current_found.add(match_result['name'])
                
                # Update verified set
                if current_found:
                    self.verified_drugs.update(current_found)
            
            else:
                time.sleep(0.01) # ถ้าไม่มีภาพใหม่ให้นอนรอแป๊บ

    def stop(self):
        self.running = False

# ==========================================
# 🎨 UI DRAWING (Clean Mode)
# ==========================================
def draw_ui(img, patient_info, found_set, fps):
    h, w = img.shape[:2]
    
    # 1. Draw FPS (Top Left)
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 2. Draw Patient Panel (Right Side)
    panel_w = 280
    panel_h = 100 + (len(patient_info['drugs']) * 30)
    x1, y1 = w - panel_w - 10, 10
    x2, y2 = w - 10, 10 + panel_h
    
    # Background
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 2)
    
    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "PATIENT INFO", (x1+10, y1+25), font, 0.6, (0, 255, 255), 2)
    cv2.line(img, (x1+10, y1+35), (x2-10, y1+35), (100, 100, 100), 1)
    cv2.putText(img, f"HN: {patient_info['hn']}", (x1+10, y1+55), font, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"{patient_info['name']}", (x1+10, y1+75), font, 0.5, (255, 255, 255), 1)
    
    start_y = y1 + 105
    for drug in patient_info['drugs']:
        is_found = False
        for found in found_set:
            if drug.lower() in found.lower() or found.lower() in drug.lower():
                is_found = True
                break
        icon = "[/]" if is_found else "[ ]"
        color = (0, 255, 0) if is_found else (150, 150, 150)
        cv2.putText(img, f"{icon} {drug}", (x1+10, start_y), font, 0.5, color, 1)
        start_y += 25

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
def main():
    print("🚀 Starting PillTrack: 60 FPS Mode...")
    
    # 1. Setup Models & Data
    def get_optimized_model_path(path):
        onnx_path = path.replace('.pt', '.onnx')
        return onnx_path if os.path.exists(onnx_path) else path

    model_path = get_optimized_model_path(config.MODEL_YOLO_PATH)
    his = HISSystem()
    patient_data = his.get_patient_info("HN001")
    patient_info = {
        "hn": "HN001",
        "name": patient_data['name'],
        "drugs": patient_data['drugs']
    }

    # 2. Start Async AI Worker
    ai_worker = AsyncDetector(model_path, patient_info['drugs']).start()

    # 3. Start Camera
    vs = WebcamStream().start()
    
    print("⏳ Waiting for camera feed...")
    while vs.read() is None:
        time.sleep(0.1)
    
    window_name = "PillTrack (60 FPS)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # FPS Calculation
    prev_time = 0
    curr_fps = 0

    try:
        while True:
            # 1. Get Frame (เร็วมาก)
            frame = vs.read()
            if frame is None: continue
            
            # 2. ส่งภาพไปให้ AI ประมวลผล (แบบไม่รอผลลัพธ์)
            ai_worker.update_frame(frame)

            # 3. เตรียมภาพสำหรับแสดงผล (Raw RGB888)
            ui_frame = frame.copy()

            # 4. ดึงผลลัพธ์ล่าสุดจาก AI มาแสดง (เฉพาะ Panel Text ไม่วาดกล่อง)
            found_drugs = ai_worker.get_verified_drugs()
            
            # คำนวณ FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # วาด UI
            draw_ui(ui_frame, patient_info, found_drugs, fps)

            # 5. Display (Smooth!)
            cv2.imshow(window_name, ui_frame)
            if cv2.waitKey(1) == ord('q'): break

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")

    finally:
        ai_worker.stop()
        vs.stop()
        cv2.destroyAllWindows()
        print("Closed Program Now")

if __name__ == "__main__":
    main()