import os

# =========================
# 🔧 SYSTEM SETTINGS
# =========================
USE_CAMERA = True                # ✅ เปิดใช้กล้อง
CAMERA_ID = 0                    # ✅ ปกติ Pi Camera จะเป็น 0
DEBUG_MODE = False               # ปิด debug เพื่อ production
DEBUG_DIR = 'debug_output'

# =========================
# 🧬 DATA PREPARATION SETTINGS
# =========================
AUGMENTATION = True              # เปิด data augmentation
AUGMENTATION_TYPES = ['flip', 'rotate', 'brightness', 'crop']
CLASS_BALANCE = True             # ทำ class balancing
SPLIT_RATIO = {'train': 0.7, 'val': 0.2, 'test': 0.1}  # สัดส่วน train/val/test
RANDOM_SEED = 42                 # เพื่อ reproducibility

# =========================
# 🧠 AI MODE SETTINGS
# =========================
USE_NEURAL_NETWORK = True        # เปิด Neural Network
USE_HYBRID_MATCHING = True       # เปิด Hybrid Matching
NEURAL_THRESHOLD = 0.7           # Threshold สำหรับ Neural Network matching
HYBRID_NN_WEIGHT = 0.7           # น้ำหนัก Neural Network ใน hybrid matching
HYBRID_SIFT_WEIGHT = 0.3         # น้ำหนัก SIFT ใน hybrid matching

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
YOLO_CONF_THRESHOLD = 0.5       # ปรับ threshold ให้เหมาะกับการเทรนจริง
MIN_DETECTION_SIZE = 20         # Minimum detection size in pixels

# =========================
# 🔍 NEURAL NETWORK SETTINGS
# =========================
NEURAL_BACKBONE = 'efficientnet_b0'    # ใช้ efficientnet_b0 สำหรับงานยา
NEURAL_FEATURE_DIM = 256        # Dimension ของ features
NEURAL_MIN_CONFIDENCE = 0.65    # ความมั่นใจขั้นต่ำสำหรับ Neural Network