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
# 🖱️ UI & STATE MANAGEMENT
# ==========================================
STATE_MENU = 0
STATE_DOUBLE_CHECK = 1
STATE_COUNTING = 2
STATE_SETTINGS = 3

current_state = STATE_MENU
active_buttons = []
last_click_time = 0
verified_drugs = set()

# ✅ ตัวแปร Zoom (1.0 = ปกติ, 2.0 = ซูม 2 เท่า)
zoom_level = 1.0 

class Button:
    def __init__(self, x, y, w, h, text, color=(50, 50, 50), text_color=(255, 255, 255), action=None):
        self.rect = (x, y, w, h)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.action = action

    def draw(self, img):
        x, y, w, h = self.rect
        cv2.rectangle(img, (x+5, y+5), (x+w+5, y+h+5), (20, 20, 20), -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), self.color, -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (200, 200, 200), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (fw, fh), _ = cv2.getTextSize(self.text, font, scale, thickness)
        tx = x + (w - fw) // 2
        ty = y + (h + fh) // 2
        cv2.putText(img, self.text, (tx, ty), font, scale, self.text_color, thickness)

    def is_clicked(self, mx, my):
        x, y, w, h = self.rect
        return x <= mx <= x+w and y <= my <= y+h

def mouse_callback(event, x, y, flags, param):
    global last_click_time
    if event == cv2.EVENT_LBUTTONDOWN:
        if time.time() - last_click_time < 0.2: return # ลด delay ลงนิดนึงให้กดรัวได้
        last_click_time = time.time()
        for btn in active_buttons:
            if btn.is_clicked(x, y):
                if btn.action: btn.action()
                return

# Actions
def go_menu(): global current_state; current_state = STATE_MENU
def go_check(): global current_state; current_state = STATE_DOUBLE_CHECK
def go_count(): global current_state; current_state = STATE_COUNTING
def go_settings(): global current_state; current_state = STATE_SETTINGS
def do_update(): print("🔄 Model Updating..."); time.sleep(0.5); print("✅ Done")

# ✅ Zoom Actions
def zoom_in():
    global zoom_level
    if zoom_level < 3.0: zoom_level += 0.5

def zoom_out():
    global zoom_level
    if zoom_level > 1.0: zoom_level -= 0.5

# ✅ ฟังก์ชัน Digital Zoom
def apply_zoom(img, scale=1.0):
    if scale <= 1.0: return img
    h, w = img.shape[:2]
    
    # คำนวณขนาดใหม่ที่จะ Crop (ยิ่งซูมเยอะ กรอบยิ่งเล็ก)
    new_w = int(w / scale)
    new_h = int(h / scale)
    
    # หาจุดกึ่งกลาง
    cx, cy = w // 2, h // 2
    
    # คำนวณพิกัด Crop
    x1 = max(0, cx - new_w // 2)
    y1 = max(0, cy - new_h // 2)
    x2 = min(w, x1 + new_w)
    y2 = min(h, y1 + new_h)
    
    # Crop และ Resize กลับมาเท่าเดิม
    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def draw_patient_panel(img, patient_info, found_set):
    h, w = img.shape[:2]
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

class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 60)
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
# 🚀 MAIN LOOP
# ==========================================
def main():
    global active_buttons, verified_drugs, zoom_level
    print("🚀 Starting PillTrack UI...")
    
    # 1. LOAD ENGINES
    if config.MODEL_YOLO_PATH.endswith('.pt'):
        main_model_path = config.MODEL_YOLO_PATH
    else:
        main_model_path = config.MODEL_YOLO_PATH.replace('.onnx', '.pt')
    
    yolo_main = YOLODetector(main_model_path)

    if config.MODEL_COUNT_PATH.endswith('.onnx'):
        count_model_path = config.MODEL_COUNT_PATH
    else:
        count_model_path = config.MODEL_COUNT_PATH.replace('.onnx', '.pt')

    if os.path.exists(count_model_path):
        yolo_counter = YOLODetector(count_model_path)
    else:
        yolo_counter = yolo_main

    identifier = SIFTIdentifier()
    db = VectorDB()
    his = HISSystem()
    
    patient_data = his.get_patient_info("HN001")
    patient_info = {
        "hn": "HN001",
        "name": patient_data['name'],
        "drugs": patient_data['drugs']
    }

    camera_source = 0
    try:
        vs = WebcamStream(src=camera_source).start()
        time.sleep(1.0)
    except Exception as e:
        print(f"❌ Camera Error: {e}")
        return

    window_name = "PillTrack"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    frame_count = 0
    SIFT_SKIP_FRAMES = 5

    try:
        while True:
            raw_frame = vs.read()
            if raw_frame is None: continue
            
            # ✅ Apply Zoom (ถ้ามีการซูม)
            if zoom_level > 1.0:
                frame = apply_zoom(raw_frame, zoom_level)
            else:
                frame = raw_frame.copy()
            
            ui_frame = frame.copy()
            h, w = ui_frame.shape[:2]
            active_buttons = []

            if current_state == STATE_MENU:
                # Reset zoom เมื่อกลับเมนู
                zoom_level = 1.0 
                
                overlay = ui_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, ui_frame, 0.3, 0, ui_frame)
                
                cv2.putText(ui_frame, "MAIN MENU", (w//2 - 100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
                cx = (w - 300) // 2
                btn1 = Button(cx, 150, 300, 80, "1. Double Check", (0, 100, 255), action=go_check)
                btn2 = Button(cx, 250, 300, 80, "2. Count Pills", (0, 200, 0), action=go_count)
                btn3 = Button(cx, 350, 300, 80, "3. Settings", (100, 100, 100), action=go_settings)
                active_buttons = [btn1, btn2, btn3]

            elif current_state == STATE_DOUBLE_CHECK:
                results = yolo_main.detect(frame, conf=0.60, iou=0.45, agnostic_nms=True, max_det=5, imgsz=320)
                
                if frame_count % SIFT_SKIP_FRAMES == 0:
                    for i, box in enumerate(results.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        if (x2-x1)*(y2-y1) < (w*h * 0.01): continue
                        
                        mask = results.masks[i] if results.masks else None
                        crop_img = yolo_main.get_crop(frame, box, mask)
                        
                        match_result = db.search(identifier, crop_img, target_drugs=patient_info['drugs'])
                        if match_result:
                            verified_drugs.add(match_result['name'])

                draw_patient_panel(ui_frame, patient_info, verified_drugs)
                back_btn = Button(20, h-70, 120, 50, "< BACK", (0, 0, 200), action=go_menu)
                active_buttons = [back_btn]

            elif current_state == STATE_COUNTING:
                # ✅ ใช้ imgsz=640 เพื่อให้เห็นของเล็กๆ ชัดขึ้น (High Res Mode)
                # ปรับ conf ลงเพื่อให้เจอง่ายขึ้นในระยะไกล
                results = yolo_counter.detect(frame, conf=0.40, iou=0.40, agnostic_nms=True, max_det=100, imgsz=320)
                
                count = len(results.boxes)
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    # วาดวงกลมเล็กๆ จะได้ไม่บังยา
                    cv2.circle(ui_frame, (cx, cy), 4, (0, 255, 255), -1)
                    cv2.circle(ui_frame, (cx, cy), 6, (0, 0, 0), 1)
                
                # --- ZOOM UI CONTROLS ---
                # ปุ่มซูมเข้า (+)
                zoom_in_btn = Button(w-70, h-140, 50, 50, "+", (0, 100, 0), action=zoom_in)
                # ปุ่มซูมออก (-)
                zoom_out_btn = Button(w-70, h-80, 50, 50, "-", (0, 0, 100), action=zoom_out)
                
                # Show Zoom Level
                cv2.putText(ui_frame, f"ZOOM: {zoom_level}x", (w-120, h-150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Show Count Display
                cv2.rectangle(ui_frame, (w-200, 20), (w-20, 100), (30, 30, 30), -1)
                cv2.putText(ui_frame, f"{count}", (w-160, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
                cv2.putText(ui_frame, "PILLS", (w-160, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                back_btn = Button(20, h-70, 120, 50, "< BACK", (0, 0, 200), action=go_menu)
                active_buttons = [back_btn, zoom_in_btn, zoom_out_btn]

            elif current_state == STATE_SETTINGS:
                overlay = ui_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (30, 30, 30), -1)
                cv2.addWeighted(overlay, 0.9, ui_frame, 0.1, 0, ui_frame)
                
                cv2.putText(ui_frame, "SETTINGS", (w//2 - 80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                update_btn = Button((w-300)//2, 200, 300, 80, "UPDATE MODEL", (200, 150, 0), action=do_update)
                back_btn = Button(20, h-70, 120, 50, "< BACK", (0, 0, 200), action=go_menu)
                active_buttons = [update_btn, back_btn]

            # Draw Buttons
            for btn in active_buttons:
                btn.draw(ui_frame)

            cv2.imshow(window_name, ui_frame)
            if cv2.waitKey(1) == ord('q'): break
            frame_count += 1

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")

    finally:
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()