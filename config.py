import os

# =========================
# 🔧 SYSTEM SETTINGS
# =========================
USE_CAMERA = True                # ✅ เปิดใช้กล้อง
CAMERA_ID = 0                    # ✅ ปกติ Pi Camera จะเป็น 0
DEBUG_MODE = True                
DEBUG_DIR = 'debug_output'
# =========================
# 📂 PATHS
# =========================
BASE_DIR = os.getcwd()
MODEL_YOLO_PATH = os.path.join(BASE_DIR, 'best_process_2.onnx')  
# MODEL_YOLO_PATH = os.path.join(BASE_DIR, 'seg_process-best.onnx')  

DB_YOLO_PATH = os.path.join(BASE_DIR,'seg_db_best.pt')
DB_FILE_PATH = os.path.join(BASE_DIR, 'drug_db.pkl')      
MODEL_COUNT_PATH = os.path.join(BASE_DIR, 'box_count_yolo.onnx')      
DEBUG_DIR = os.path.join(BASE_DIR, 'debug_output')

# =========================
# 🧠 AI THRESHOLDS (Logic เดิม)
# =========================
SIFT_MIN_MATCH_COUNT = 14         # ⬅️ กลับมาเป็น 8 (หาง่าย)
SIFT_MATCH_RATIO = 0.75          # ⬅️ กลับมาเป็น 0.75