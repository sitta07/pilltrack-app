import os

# =========================
# 🔧 SYSTEM SETTINGS
# =========================
USE_CAMERA = True                # ✅ เปิดใช้กล้อง
CAMERA_ID = 0                    # ✅ ปกติ Pi Camera จะเป็น 0
DEBUG_MODE = True                
DEBUG_DIR = 'debug_output'

# =========================
# 🧠 AI MODE SETTINGS
# =========================
USE_NEURAL_NETWORK = False       # ✅ ปิดชั่วคราวเพื่อ debug
USE_HYBRID_MATCHING = False      # ✅ ใช้ SIFT เท่านั้นก่อน
NEURAL_THRESHOLD = 0.75         # ✅ Threshold สำหรับ Neural Network matching
HYBRID_NN_WEIGHT = 0.7          # ✅ น้ำหนัก Neural Network ใน hybrid matching
HYBRID_SIFT_WEIGHT = 0.3        # ✅ น้ำหนัก SIFT ใน hybrid matching

# =========================
# 📂 PATHS
# =========================
BASE_DIR = os.getcwd()
MODEL_YOLO_PATH = os.path.join(BASE_DIR, 'best_process_2.onnx')  
DB_YOLO_PATH = os.path.join(BASE_DIR, 'seg_db_best.pt')
DB_FILE_PATH = os.path.join(BASE_DIR, 'drug_db.pkl')
NEURAL_DB_FILE_PATH = os.path.join(BASE_DIR, 'drug_db_nn.pkl')  # ✅ Neural Database
NEURAL_MODEL_PATH = os.path.join(BASE_DIR, 'neural_model.pt')   # ✅ Neural Network Model
MODEL_COUNT_PATH = os.path.join(BASE_DIR, 'box_count_yolo.onnx')
DEBUG_DIR = os.path.join(BASE_DIR, 'debug_output')

# =========================
# 🧠 AI THRESHOLDS
# =========================
SIFT_MIN_MATCH_COUNT = 12
SIFT_MATCH_RATIO = 0.75

# =========================
# 🔍 NEURAL NETWORK SETTINGS
# =========================
NEURAL_BACKBONE = 'resnet18'    # ✅ 'resnet18', 'mobilenet_v2', 'efficientnet_b0'
NEURAL_FEATURE_DIM = 256        # ✅ Dimension ของ features
NEURAL_MIN_CONFIDENCE = 0.60    # ✅ ความมั่นใจขั้นต่ำสำหรับ Neural Network