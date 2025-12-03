from ultralytics import YOLO
import config

# โหลดโมเดลตามที่ Config ชี้อยู่
model_path = config.MODEL_YOLO_PATH.replace('.onnx', '.pt')
print(f"🧐 Inspecting Model: {model_path}")

try:
    model = YOLO(model_path)
    print("\n✅ Load Success!")
    print("---------------------------------------------------")
    print(f"📂 Class Names Map: {model.names}")
    print("---------------------------------------------------")
    
    # เช็คว่ามีคำว่า panel หรือ box ไหม
    names = model.names.values()
    print(f"👉 Has 'panel'? : {'panel' in str(names)}")
    print(f"👉 Has 'box'?   : {'box' in str(names)}")

except Exception as e:
    print(f"❌ Error loading model: {e}")