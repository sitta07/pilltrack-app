import cv2
import time
import os
import numpy as np
import threading
import sys

# ✅ FIX Display on Raspberry Pi OS
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
# 🌡️ UTILS
# ==========================================
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read()) / 1000.0
    except:
        return 0.0

# ==========================================
# 📷 WEBCAM STREAM (HD 720p)
# ==========================================
class WebcamStream:
    def __init__(self):
        self.stopped = False
        self.frame = None
        self.grabbed = False
        self.picam2 = None

    def start(self):
        print("📷 Initializing Picamera2 (HD Mode)...")
        try:
            self.picam2 = Picamera2()
            # HD 1280x720 @ 60FPS
            config = self.picam2.create_preview_configuration(
                main={"size": (1280, 720), "format": "RGB888"},
                controls={"FrameDurationLimits": (16666, 16666)} 
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(2.0)
            print("✅ Camera Ready!")
        except Exception as e:
            print(f"❌ Camera Init Failed: {e}")
            self.stopped = True
            
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                frame = self.picam2.capture_array()
                if frame is not None:
                    self.frame = frame
                    self.grabbed = True
                else:
                    self.stopped = True
            except:
                self.stopped = True

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
# 🧠 ASYNC AI WORKER
# ==========================================
class AsyncDetector:
    def __init__(self, model_path, patient_drugs):
        self.yolo = YOLODetector(model_path)
        self.identifier = SIFTIdentifier()
        self.db = VectorDB()
        self.patient_drugs = patient_drugs
        
        self.latest_frame = None
        self.verified_drugs = set()
        self.latest_boxes = [] # เก็บกล่องเพื่อส่งไปหาจุดกึ่งกลาง
        self.running = True
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()
        return self

    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()

    def get_results(self):
        return self.verified_drugs, self.latest_boxes

    def run(self):
        print("🧠 AI Worker Running...")
        while self.running:
            frame_to_process = None
            
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None

            if frame_to_process is not None:
                h, w = frame_to_process.shape[:2]
                
                # 1. YOLO Detect
                # เพิ่ม conf เป็น 0.65 เพื่อลดการตรวจจับขยะ (ลดจุดมั่ว)
                results = self.yolo.detect(frame_to_process, conf=0.65, iou=0.5, agnostic_nms=True, max_det=5, imgsz=320)
                
                detected_boxes = []
                valid_boxes_for_sift = []

                # 2. Process Boxes
                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    area = (x2-x1)*(y2-y1)
                    
                    # เก็บ Coordinates ส่งให้ UI
                    detected_boxes.append((x1, y1, x2, y2))
                    
                    if area > (w*h * 0.02): 
                        valid_boxes_for_sift.append((area, box, i))
                
                # ส่งข้อมูลกล่องไปให้ UI ทันที
                self.latest_boxes = detected_boxes

                # 3. SIFT (เฉพาะเม็ดใหญ่สุด)
                valid_boxes_for_sift.sort(key=lambda x: x[0], reverse=True)
                target_boxes = valid_boxes_for_sift[:1] 

                current_found = set()
                for _, box, idx in target_boxes:
                    mask = results.masks[idx] if results.masks else None
                    crop_img = self.yolo.get_crop(frame_to_process, box, mask)
                    match_result = self.db.search(self.identifier, crop_img, target_drugs=self.patient_drugs)
                    if match_result:
                        current_found.add(match_result['name'])
                
                if current_found:
                    self.verified_drugs.update(current_found)
            
            else:
                time.sleep(0.01)

    def stop(self):
        self.running = False

# ==========================================
# 🎨 UI DRAWING (Dots Only)
# ==========================================
def draw_ui(img, patient_info, found_set, boxes, fps):
    h, w = img.shape[:2]
    
    # ✅ 1. Draw DOTS instead of Rectangles
    for (x1, y1, x2, y2) in boxes:
        # หาจุดกึ่งกลาง
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        # วาดจุด (วงกลมทึบ)
        # สีเขียวอ่อน (Lime Green)
        cv2.circle(img, (cx, cy), 6, (50, 255, 50), -1) 
        # วาดขอบดำบางๆ รอบจุดเพื่อให้เห็นชัดขึ้น
        cv2.circle(img, (cx, cy), 7, (0, 0, 0), 1)

    # 2. FPS & Temp
    temp = get_cpu_temperature()
    temp_color = (0, 255, 0) if temp < 80 else (255, 0, 0)
    
    cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(img, f"TEMP: {temp:.1f} C", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, temp_color, 2)

    # 3. Patient Info Panel
    panel_w = 300
    panel_h = 100 + (len(patient_info['drugs']) * 35)
    x1, y1 = w - panel_w - 20, 20
    x2, y2 = w - 20, 20 + panel_h
    
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "PATIENT INFO", (x1+10, y1+30), font, 0.7, (0, 255, 255), 2)
    cv2.line(img, (x1+10, y1+40), (x2-10, y1+40), (100, 100, 100), 1)
    cv2.putText(img, f"HN: {patient_info['hn']}", (x1+10, y1+65), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"{patient_info['name']}", (x1+10, y1+90), font, 0.6, (255, 255, 255), 1)
    
    start_y = y1 + 125
    for drug in patient_info['drugs']:
        is_found = False
        for found in found_set:
            if drug.lower() in found.lower() or found.lower() in drug.lower():
                is_found = True
                break
        icon = "[/]" if is_found else "[ ]"
        color = (0, 255, 0) if is_found else (150, 150, 150)
        cv2.putText(img, f"{icon} {drug}", (x1+10, start_y), font, 0.6, color, 1)
        start_y += 30

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
def main():
    print("🚀 Starting PillTrack (Dot Mode)...")
    
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

    ai_worker = AsyncDetector(model_path, patient_info['drugs']).start()
    vs = WebcamStream().start()
    
    print("⏳ Waiting for camera...")
    while vs.read() is None:
        time.sleep(0.1)
    
    window_name = "PillTrack"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_time = 0
    
    try:
        while True:
            frame = vs.read()
            if frame is None: continue
            
            ai_worker.update_frame(frame)
            found_drugs, boxes = ai_worker.get_results()
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            ui_frame = frame.copy()
            draw_ui(ui_frame, patient_info, found_drugs, boxes, fps)

            cv2.imshow(window_name, ui_frame)
            if cv2.waitKey(1) == ord('q'): break
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