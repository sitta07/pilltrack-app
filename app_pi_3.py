import cv2
import time
import os
import numpy as np
import threading
import sys

# ✅ FIX Display on Raspberry Pi OS
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
# 🌡️ UTILS: CPU TEMPERATURE
# ==========================================
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read()) / 1000.0
        return temp
    except:
        return 0.0

# ==========================================
# 📷 WEBCAM STREAM (HD 720p @ 60FPS)
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
            
            # ✅ ปรับเป็น 1280x720 (HD)
            # ภาพบนจอจะชัดขึ้นมาก แต่ยังรักษา 60 FPS ไหวบน Pi 5
            config = self.picam2.create_preview_configuration(
                main={"size": (1280, 720), "format": "RGB888"},
                controls={"FrameDurationLimits": (16666, 16666)} 
            )
            self.picam2.configure(config)
            self.picam2.start()
            
            time.sleep(2.0)
            print("✅ Camera Ready (1280x720)!")
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
        self.running = True
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()
        return self

    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()

    def get_verified_drugs(self):
        return self.verified_drugs

    def run(self):
        print("🧠 AI Worker Running...")
        while self.running:
            frame_to_process = None
            
            with self.lock:
                if self.latest_frame is not None:
                    # ตรงนี้เรารับภาพ 1280x720 มา
                    frame_to_process = self.latest_frame
                    self.latest_frame = None

            if frame_to_process is not None:
                h, w = frame_to_process.shape[:2]
                
                # 🟢 START: LOGIC กรองยาที่เจอแล้ว
                # คำนวณยาที่เหลือที่ต้องหา: ยาทั้งหมด - ยาที่เจอแล้ว
                drugs_to_find = list(set(self.patient_drugs) - self.verified_drugs)
                
                # ถ้าหายาครบแล้ว (ไม่ต้องรัน SIFT ต่อ)
                if not drugs_to_find and self.verified_drugs:
                    time.sleep(0.1) 
                    continue
                # 🟢 END: LOGIC กรองยาที่เจอแล้ว
                
                # 1. YOLO Detect
                # imgsz=320 สำคัญมาก! มันบอก YOLO ว่า "ย่อภาพให้เหลือ 320 นะก่อนตรวจ"
                results = self.yolo.detect(frame_to_process, conf=0.5, iou=0.45, agnostic_nms=True, max_det=5, imgsz=320)
                
                # 2. Sort Boxes
                valid_boxes = []
                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    area = (x2-x1)*(y2-y1)
                    if area > (w*h * 0.02): 
                        valid_boxes.append((area, box, i))
                
                valid_boxes.sort(key=lambda x: x[0], reverse=True)
                target_boxes = valid_boxes[:1] 

                current_found = set()
                # 3. SIFT Logic
                for _, box, idx in target_boxes:
                    mask = results.masks[idx] if results.masks else None
                    # Crop ภาพยาจากภาพ HD (ทำให้ SIFT เห็นลายละเอียดชัดขึ้นด้วย!)
                    crop_img = self.yolo.get_crop(frame_to_process, box, mask)
                    
                    # 🟢 CHANGE: ส่งเฉพาะ drugs_to_find ไปให้ SIFT เทียบ
                    match_result = self.db.search(self.identifier, crop_img, target_drugs=drugs_to_find)
                    if match_result:
                        current_found.add(match_result['name'])
                
                if current_found:
                    self.verified_drugs.update(current_found)
            
            else:
                time.sleep(0.01)

    def stop(self):
        self.running = False

# ==========================================
# 🎨 UI DRAWING
# ==========================================
def draw_ui(img, patient_info, found_set, fps):
    h, w = img.shape[:2]
    
    # 1. Draw FPS & Temp (Top Left)
    temp = get_cpu_temperature()
    temp_color = (0, 255, 0) if temp < 80 else (255, 0, 0)
    
    # ปรับขนาดตัวอักษรนิดหน่อยให้เข้ากับจอ HD
    cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(img, f"TEMP: {temp:.1f} C", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, temp_color, 2)

    # 2. Patient Info Panel (ปรับตำแหน่งให้ชิดขวาของจอ HD)
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
        # ตรวจสอบว่าชื่อยาในลิสต์ผู้ป่วยตรงกับยาที่เจอแล้วหรือไม่
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
    print("🚀 Starting PillTrack (HD 720p Mode)...")
    
    # 1. Setup
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

    # 2. Workers
    ai_worker = AsyncDetector(model_path, patient_info['drugs']).start()
    vs = WebcamStream().start()
    
    print("⏳ Waiting for camera feed...")
    while vs.read() is None:
        time.sleep(0.1)
    
    window_name = "PillTrack"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # คำสั่งนี้จะดึงภาพ HD ให้เต็มจอ Monitor อัตโนมัติ
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_time = 0
    
    try:
        while True:
            frame = vs.read()
            if frame is None: continue
            
            # ส่งภาพ HD ไปให้ AI (AI จะย่อเองภายใน)
            ai_worker.update_frame(frame)

            # รับผลลัพธ์
            found_drugs = ai_worker.get_verified_drugs()
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # วาด UI บนภาพ HD
            ui_frame = frame.copy()
            draw_ui(ui_frame, patient_info, found_drugs, fps)

            # แสดงผล
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