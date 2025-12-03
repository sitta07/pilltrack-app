import pickle
import os
import cv2
import config

class VectorDB:
    def __init__(self, db_path=config.DB_FILE_PATH):
        self.db_path = db_path
        self.data = []
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            print(f"📂 Loading Database from {self.db_path}")
            with open(self.db_path, 'rb') as f:
                raw_data = pickle.load(f)
                self.data = []
                for item in raw_data:
                    # Deserialize Keypoints
                    kp_restored = [
                        cv2.KeyPoint(x=p[0][0], y=p[0][1], size=p[1], angle=p[2], 
                                     response=p[3], octave=p[4], class_id=p[5]) 
                        for p in item['kp']
                    ]
                    self.data.append({
                        'name': item['name'],
                        'kp': kp_restored,
                        'des': item['des']
                        # ❌ ไม่โหลด hist
                        # ❌ ไม่โหลด type
                    })
        else:
            print("⚠️ Database file not found!")

    # ✅ search แบบดั้งเดิม (รับแค่ target_drugs ไม่รับ use_color)
    def search(self, identifier_engine, query_img, target_drugs=None, **kwargs):
        # 1. Extract Feature (รับแค่ 2 ค่า)
        kp_q, des_q = identifier_engine.extract_features(query_img)
        query_pack = (kp_q, des_q)
        
        best_match = None
        best_score = 0
        
        for item in self.data:
            # Filter ชื่อยา
            if target_drugs and item['name'] not in target_drugs:
                continue
            
            # Compare SIFT
            db_pack = (item['kp'], item['des'])
            inliers = identifier_engine.compare(query_pack, db_pack)
            
            if inliers >= config.SIFT_MIN_MATCH_COUNT:
                if inliers > best_score:
                    best_score = inliers
                    best_match = {
                        'name': item['name'],
                        'inliers': inliers
                    }
        
        return best_match