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
    from engines import SIFTIdentifier
    from database import VectorDB
    from his_mock import HISSystem
    from picamera2 import Picamera2
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# ==========================================
# ⚡ UNIVERSAL FAST YOLO (Auto-Fix Shape)
# ==========================================
class FastYOLODetector:
    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.4):
        print(f"⚡ Loading FastYOLO (Universal Mode): {model_path}")
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = (320, 320) # ต้องตรงกับตอน export (แนะนำ 320 เพื่อความเร็ว)

    def detect(self, image):
        # 1. Prepare Input
        blob = cv2.dnn.blobFromImage(image, 1/255.0, self.input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # 2. Inference
        # OpenCV บางรุ่นคืนค่าเป็น tuple (outputs, layerNames)
        raw_output = self.net.forward()

        # 3. ✅ AUTO-FIX SHAPE LOGIC (แก้ปัญหา outputs[0] error)
        # ตรวจสอบว่า raw_output เป็น list/tuple หรือ numpy array
        if isinstance(raw_output, (list, tuple)):
            predictions = raw_output[0]
        else:
            predictions = raw_output

        # ลดมิติ Batch ออก: (1, 84, 8400) -> (84, 8400)
        predictions = np.squeeze(predictions)

        # ตรวจสอบว่าต้อง Transpose ไหม? 
        # YOLOv8 ปกติจะมาแบบ (Channels, Anchors) เช่น (6, 8400)
        # เราต้องการ (Anchors, Channels) เช่น (8400, 6) เพื่อวนลูป
        if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.transpose()

        # 4. Extract Boxes
        boxes = []
        scores = []
        class_ids = []
        
        # Scaling Factors
        img_h, img_w = image.shape[:2]
        x_scale = img_w / self.input_size[0]
        y_scale = img_h / self.input_size[1]

        # วนลูปหา object (predictions คือ array ขนาด [8400, 4+classes])
        # เร่งความเร็วด้วยการกรอง confidence ก่อนวนลูป numpy
        
        # หา max score ของแต่ละ row
        # (YOLOv8: 0-3 คือ bbox, 4 เป็นต้นไปคือ class scores)
        if predictions.shape[1] > 4:
            class_scores = predictions[:, 4:]
            max_scores = np.max(class_scores, axis=1)
            max_indices = np.argmax(class_scores, axis=1)
            
            # กรองเฉพาะอันที่มั่นใจ (Vectorized Operation เร็วปรื๋อ)
            keep_indices = max_scores >= self.conf_thres
            
            filtered_preds = predictions[keep_indices]
            filtered_scores = max_scores[keep_indices]
            filtered_classes = max_indices[keep_indices]
            
            for i, pred in enumerate(filtered_preds):
                # YOLO format: cx, cy, w, h
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
                
                # แปลงเป็น Pixel จริง
                left = int((cx - w/2) * x_scale)
                top = int((cy - h/2) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                
                boxes.append([left, top, width, height])
                scores.append(float(filtered_scores[i]))
                class_ids.append(filtered_classes[i])

        # 5. NMS (แก้กล่องซ้อน)
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        
        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                # Clip ให้อยู่ในหน้าจอ
                x = max(0, x)
                y = max(0, y)
                final_boxes.append((x, y, x+w, y+h))
                
        return final_boxes

    def get_crop(self, frame, box):
        x1, y1, x2, y2 = box
        return frame[y1:y2, x1:x2]

# ==========================================
# 📷 WEBCAM STREAM
# ==========================================
class WebcamStream:
    def __init__(self):
        self.stopped = False
        self.frame = None
        self.picam2 = None

    def start(self):
        print("📷 Camera: HD 720p Mode")
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (1280, 720), "format": "RGB888"},
                controls={"FrameDurationLimits": (16666, 16666)}
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(2.0)
        except Exception as e:
            print(f"❌ Camera Error: {e}")
            self.stopped = True
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
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
        self.yolo = FastYOLODetector(model_path, conf_thres=0.6, iou_thres=0.5)
        self.identifier = SIFTIdentifier()
        self.db = VectorDB()
        self.patient_drugs = patient_drugs
        
        self.latest_frame = None
        self.verified_drugs = set()
        self.latest_boxes = []
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
        print("🧠 Fast AI Worker Running...")
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None

            if frame_to_process is not None:
                h, w = frame_to_process.shape[:2]
                
                # 1. Detect
                boxes = self.yolo.detect(frame_to_process)
                
                # 2. Filter & Sort
                valid_boxes = []
                self.latest_boxes = boxes # Send to UI
                
                for box in boxes:
                    x1, y1, x2, y2 = box
                    area = (x2-x1)*(y2-y1)
                    if area > (w*h * 0.01): # Min Area 1%
                         valid_boxes.append((area, box))
                
                valid_boxes.sort(key=lambda x: x[0], reverse=True)
                target_boxes = valid_boxes[:1]

                # 3. SIFT
                current_found = set()
                for _, box in target_boxes:
                    crop_img = self.yolo.get_crop(frame_to_process, box)
                    match_result = self.db.search(self.identifier, crop_img, target_drugs=self.patient_drugs)
                    if match_result:
                        current_found.add(match_result['name'])
                
                if current_found:
                    self.verified_drugs.update(current_found)
            else:
                time.sleep(0.01)

    def stop(self): self.running = False

# ==========================================
# 🎨 UI DRAWING
# ==========================================
def draw_ui(img, patient_info, found_set, boxes, fps):
    h, w = img.shape[:2]
    
    # Dots
    for (x1, y1, x2, y2) in boxes:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(img, (cx, cy), 6, (50, 255, 50), -1)
        cv2.circle(img, (cx, cy), 7, (0, 0, 0), 1)

    # Info
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
    print("🚀 Starting PillTrack (Auto-Fix Shape Mode)...")
    
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
            
            ai_worker.update_frame(frame)
            found, boxes = ai_worker.get_results()
            
            curr = time.time()
            fps = 1/(curr-prev_time) if curr>prev_time else 0
            prev_time = curr
            
            ui = frame.copy()
            draw_ui(ui, p_info, found, boxes, fps)
            
            cv2.imshow("PillTrack", ui)
            if cv2.waitKey(1) == ord('q'): break
            time.sleep(0.01)

    except KeyboardInterrupt: pass
    finally:
        ai_worker.stop()
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()