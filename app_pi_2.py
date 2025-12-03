import cv2
import time
import os
import numpy as np
import threading
import sys

# ✅ 1. FIX Display on Raspberry Pi OS (Wayland)
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Import Custom Modules
try:
    import config
    from engines import YOLODetector, SIFTIdentifier
    from database import VectorDB
    from his_mock import HISSystem
    # บังคับใช้ Picamera2
    from picamera2 import Picamera2
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)

# ==========================================
# 📷 WEBCAM STREAM (60 FPS / RGB888)
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
            
            # ✅ Config: Force 60 FPS & RGB888
            # FrameDurationLimits: (16666, 16666) microseconds = 1/60 sec
            config = self.picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                controls={"FrameDurationLimits": (16666, 16666)} 
            )
            self.picam2.configure(config)
            self.picam2.start()
            
            time.sleep(2.0) # Warm up
            print("✅ Camera Running @ 60 FPS (RGB888)!")
            
        except Exception as e:
            print(f"❌ Camera Init Failed: {e}")
            self.stopped = True
            
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                # Capture อย่างต่อเนื่อง
                frame = self.picam2.capture_array()
                if frame is not None:
                    self.frame = frame
                    self.grabbed = True
                else:
                    self.stopped = True
            except:
                self.stopped = True
            # ไม่ Sleep ตรงนี้ เพื่อรีด FPS ให้สุด

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
# 🧠 ASYNC AI WORKER (Fast Logic)
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
        # อัปเดตภาพล่าสุดเข้า Buffer
        with self.lock:
            self.latest_frame = frame.copy()

    def get_verified_drugs(self):
        return self.verified_drugs

    def run(self):
        print("🧠 AI Worker Started (Smart Focus)...")
        while self.running:
            frame_to_process = None
            
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None # Clear buffer

            if frame_to_process is not None:
                h, w = frame_to_process.shape[:2]
                
                # 1. YOLO Detect
                results = self.yolo.detect(frame_to_process, conf=0.6, iou=0.45, agnostic_nms=True, max_det=5, imgsz=320)
                
                # 2. Smart Sorting: หา "ยาเม็ดใหญ่ที่สุด" (Focus Object)
                valid_boxes = []
                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    area = (x2-x1)*(y2-y1)
                    # ต้องใหญ่กว่า 2% ของจอ ถึงจะตรวจ (กัน Noise)
                    if area > (w*h * 0.02): 
                        valid_boxes.append((area, box, i))
                
                # เรียงจากใหญ่ -> เล็ก แล้วเอาแค่ 1 อันดับแรก
                valid_boxes.sort(key=lambda x: x[0], reverse=True)
                target_boxes = valid_boxes[:1] 

                current_found = set()
                
                # 3. SIFT Verification (เฉพาะตัวที่คัดมา)
                for _, box, idx in target_boxes:
                    mask = results.masks[idx] if results.masks else None
                    crop_img = self.yolo.get_crop(frame_to_process, box, mask)
                    
                    match_result = self.db.search(self.identifier, crop_img, target_drugs=self.patient_drugs)
                    if match_result:
                        current_found.add(match_result['name'])
                
                if current_found:
                    self.verified_drugs.update(current_found)
            
            else:
                # ถ้าไม่มีภาพใหม่ ให้นอนรอนานหน่อย คืน CPU ให้ Main Thread
                time.sleep(0.01)

    def stop(self):
        self.running = False

# ==========================================
# 🎨 UI DRAWING
# ==========================================
def draw_patient_panel(img, patient_info, found_set, fps):
    h, w = img.shape[:2]
    
    # Draw FPS
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Patient Info Panel
    panel_w = 280
    panel_h = 100 + (len(patient_info['drugs']) * 30)
    x1, y1 = w - panel_w - 10, 10
    x2, y2 = w - 10, 10 + panel_h
    
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 2)
    
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
    print("🚀 Starting PillTrack: Double Check Mode (High Performance)...")
    
    # 1. Setup Models (Auto ONNX)
    def get_optimized_model_path(path):
        onnx_path = path.replace('.pt', '.onnx')
        if os.path.exists(onnx_path):
            print(f"⚡ Using ONNX Model: {onnx_path}")
            return onnx_path
        return path

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
    
    window_name = "PillTrack"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_time = 0
    
    try:
        while True:
            # 1. Get RAW Frame (Fast)
            frame = vs.read()
            if frame is None: continue
            
            # 2. Send to AI (Non-blocking)
            ai_worker.update_frame(frame)

            # 3. Prepare UI Frame
            ui_frame = frame.copy() # RGB888 Direct

            # 4. Get AI Results & Draw UI
            found_drugs = ai_worker.get_verified_drugs()
            
            # Calc FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # Draw (No Boxes, Just Panel)
            draw_patient_panel(ui_frame, patient_info, found_drugs, fps)

            # 5. Display
            cv2.imshow(window_name, ui_frame)
            
            if cv2.waitKey(1) == ord('q'): break
            
            # ✅ IMPORTANT: Sleep 0.01s เพื่อให้ CPU ว่างไปประมวลผล AI บ้าง
            # (ถ้าไม่ใส่ตรงนี้ UI จะลื่นแต่ AI จะอืด)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")

    finally:
        ai_worker.stop()
        vs.stop()
        cv2.destroyAllWindows()
        print("👋 Bye Bye!")

if __name__ == "__main__":
    main()