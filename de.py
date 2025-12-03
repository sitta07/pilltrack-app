import cv2
import config
from ultralytics import YOLO
import os

def main():
    print("🚀 Starting Simple Debugger...")

    # 1. โหลดโมเดล (ใช้ .pt ชัวร์สุด)
    if config.MODEL_YOLO_PATH.endswith('.onnx'):
        model_path = config.MODEL_YOLO_PATH.replace('.onnx', '.pt')
    else:
        model_path = config.MODEL_YOLO_PATH
    
    print(f"🔹 Loading Model: {model_path}")
    model = YOLO(model_path)

    # ปริ้นชื่อคลาสที่โมเดลรู้จักออกมาดู
    print(f"📋 Model Classes: {model.names}")

    # 2. เปิดกล้อง (แก้เลข 0 หรือ 1 ตามเครื่องคุณ)
    cap = cv2.VideoCapture(1) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 3. Detect แบบดิบๆ (Conf ต่ำๆ)
        results = model(frame, conf=0.25, verbose=False)

        # 4. วาดทุกอย่างที่เจอ
        for result in results:
            for box in result.boxes:
                # พิกัด
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # ชื่อคลาส
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])

                # วาดกรอบแดง
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # เขียนชื่อคลาสตัวใหญ่ๆ
                label = f"{class_name} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Debug View", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()