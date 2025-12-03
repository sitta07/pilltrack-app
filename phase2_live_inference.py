"""
Phase 2: Live Inference Pipeline
==================================

Real-time drug identification from camera feed:
1. Frame capture (30 FPS)
2. YOLO segmentation
3. Preprocessing
4. Feature extraction
5. FAISS search
6. Confidence decision
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import faiss
import json
import asyncio
import threading
import queue
from collections import deque
from typing import Dict, List, Tuple, Optional
import logging
import time
from dataclasses import dataclass
import timm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 300
FEATURE_DIM = 1536

@dataclass
class DrugResult:
    """Result of drug identification"""
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    status: str  # 'ACCEPT', 'PARTIAL_MATCH', 'UNKNOWN'
    top_candidates: List[str] = None
    processing_time_ms: float = 0.0


class AsyncFrameCapture:
    """Async frame capture with buffer for smooth 30 FPS"""
    
    def __init__(self, source: int = 0, buffer_size: int = 2):
        """
        Args:
            source: Camera source (0 for default)
            buffer_size: Frame buffer size
        """
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.latest_frame = None
        self.stopped = False
        self.fps_counter = deque(maxlen=30)
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, 
                                              daemon=True)
        self.capture_thread.start()
        
        logger.info(f"✅ Initialized camera capture (source={source})")
    
    def _capture_loop(self):
        """Continuous frame capture"""
        last_time = time.time()
        
        while not self.stopped:
            ret, frame = self.cap.read()
            
            if not ret:
                continue
            
            self.latest_frame = frame
            
            # FPS calculation
            current_time = time.time()
            self.fps_counter.append(current_time - last_time)
            last_time = current_time
            
            # Keep only latest frame in buffer
            if self.frame_buffer.full():
                try:
                    self.frame_buffer.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_buffer.put(frame)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame"""
        try:
            return self.frame_buffer.get_nowait()
        except queue.Empty:
            return self.latest_frame
    
    def get_fps(self) -> float:
        """Get current FPS"""
        if len(self.fps_counter) > 1:
            avg_time = np.mean(list(self.fps_counter))
            return 1.0 / avg_time if avg_time > 0 else 0
        return 0
    
    def stop(self):
        """Stop capture"""
        self.stopped = True
        self.cap.release()


class YOLODrugDetector:
    """Detect and segment drugs using YOLO"""
    
    def __init__(self, model_path: str = 'best_process_2.onnx'):
        """
        Args:
            model_path: Path to YOLO model
        """
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        logger.info(f"✅ Loaded YOLO model from {model_path}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect drugs in frame
        
        Args:
            frame: Input frame (H×W×3)
            
        Returns:
            List of detections with:
            - crop: Cropped drug image
            - bbox: (x1, y1, x2, y2)
            - confidence: Detection confidence
        """
        results = self.model(frame, verbose=False)
        
        detections = []
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Ensure bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            crop = frame[y1:y2, x1:x2]
            
            detections.append({
                'crop': crop,
                'bbox': (x1, y1, x2, y2),
                'confidence': float(box.conf),
                'class': int(box.cls)
            })
        
        return detections


class DrugPreprocessor:
    """Preprocess detected drug crops"""
    
    @staticmethod
    def preprocess(crop: np.ndarray) -> np.ndarray:
        """
        Resize to 300×300 with white padding
        
        Args:
            crop: Cropped drug image
            
        Returns:
            Preprocessed 300×300 image
        """
        h, w = crop.shape[:2]
        
        # Calculate new size (maintain aspect ratio)
        if h > w:
            new_h, new_w = IMAGE_SIZE, int(IMAGE_SIZE * w / h)
        else:
            new_h, new_w = int(IMAGE_SIZE * h / w), IMAGE_SIZE
        
        # Resize
        resized = cv2.resize(crop, (new_w, new_h), 
                            interpolation=cv2.INTER_LINEAR)
        
        # White padding
        top = (IMAGE_SIZE - new_h) // 2
        bottom = IMAGE_SIZE - new_h - top
        left = (IMAGE_SIZE - new_w) // 2
        right = IMAGE_SIZE - new_w - left
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
        return padded
    
    @staticmethod
    def normalize(image: np.ndarray) -> torch.Tensor:
        """
        Normalize and convert to tensor
        
        Args:
            image: Preprocessed image (300×300)
            
        Returns:
            Normalized tensor (1, 3, 300, 300)
        """
        image = image.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).to(DEVICE)
        
        return tensor.float()


class BatchFeatureExtractor:
    """Extract features from drug batch"""
    
    def __init__(self, model_name: str = 'efficientnet_b3'):
        """Initialize EfficientNet-B3"""
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        logger.info(f"✅ Loaded {model_name} for feature extraction")
    
    def extract_batch(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Extract features from batch of images
        
        Args:
            images: List of (1, 3, 300, 300) tensors
            
        Returns:
            (N, 1536) feature vectors, L2-normalized
        """
        batch = torch.cat(images, dim=0)
        
        with torch.no_grad():
            features = self.model(batch)
            
            # Global average pooling if needed
            if len(features.shape) == 4:
                features = F.adaptive_avg_pool2d(features, 1)
                features = features.squeeze(-1).squeeze(-1)
            
            # Ensure 1536-dim
            if features.shape[-1] != FEATURE_DIM:
                if features.shape[-1] > FEATURE_DIM:
                    features = features[:, :FEATURE_DIM]
                else:
                    pad_size = FEATURE_DIM - features.shape[-1]
                    features = F.pad(features, (0, pad_size))
            
            # L2 normalize
            features = F.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()


class FAISSSearcher:
    """Search FAISS index for drug matches"""
    
    def __init__(self, index_path: str, metadata_path: str):
        """
        Load FAISS index and metadata
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata JSON
        """
        # Load index
        self.index = faiss.read_index(index_path)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("✅ FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Could not move to GPU: {e}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        logger.info(f"✅ Loaded FAISS index with {self.metadata['num_drugs']} drugs")
    
    def search(self, query_feature: np.ndarray, 
               k: int = 5) -> Dict:
        """
        Search for similar drugs
        
        Args:
            query_feature: (1536,) normalized feature vector
            k: Number of results
            
        Returns:
            {
                'names': [name1, name2, ...],
                'scores': [score1, score2, ...],
                'top_1': best match,
                'top_1_conf': confidence score
            }
        """
        # Reshape for FAISS
        query = query_feature.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query, k)
        
        # Convert to results
        names = [self.metadata['drug_names'][idx] for idx in indices[0]]
        scores = distances[0].tolist()
        
        return {
            'names': names,
            'scores': scores,
            'top_1': names[0],
            'top_1_conf': float(scores[0]),
            'top_3': names[:3]
        }


class ConfidenceDecisionMaker:
    """Make confidence-based decisions"""
    
    THRESHOLD_HIGH = 0.85
    THRESHOLD_LOW = 0.70
    
    @staticmethod
    def decide(result: Dict) -> DrugResult:
        """
        Make decision based on confidence
        
        Args:
            result: Search result with confidence score
            
        Returns:
            DrugResult with status
        """
        confidence = result['top_1_conf']
        
        if confidence > ConfidenceDecisionMaker.THRESHOLD_HIGH:
            status = 'ACCEPT'
        elif confidence > ConfidenceDecisionMaker.THRESHOLD_LOW:
            status = 'PARTIAL_MATCH'
        else:
            status = 'UNKNOWN'
        
        return {
            'status': status,
            'confidence': confidence,
            'top_1': result['top_1'],
            'top_3': result['top_3']
        }


class TestTimeAugmentation:
    """Test-time augmentation for uncertain cases"""
    
    def __init__(self, feature_extractor: BatchFeatureExtractor,
                 searcher: FAISSSearcher):
        """
        Args:
            feature_extractor: Feature extraction model
            searcher: FAISS searcher
        """
        self.extractor = feature_extractor
        self.searcher = searcher
    
    def augment_and_search(self, image: np.ndarray) -> Dict:
        """
        Create augmented versions and vote
        
        Args:
            image: Preprocessed 300×300 image
            
        Returns:
            Consensus result with voting
        """
        # Version 1: Original
        version1 = image.copy()
        
        # Version 2: Slight rotation
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 5, 1.0)
        version2 = cv2.warpAffine(image, M, (w, h), 
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        
        # Version 3: Brightness adjustment
        version3 = cv2.convertScaleAbs(image, alpha=1.1, beta=0)
        version3 = np.clip(version3, 0, 255).astype(np.uint8)
        
        # Extract features
        versions = [version1, version2, version3]
        tensors = [DrugPreprocessor.normalize(v) for v in versions]
        features_batch = self.extractor.extract_batch(tensors)
        
        # Search each
        results = []
        confidences = []
        
        for features in features_batch:
            result = self.searcher.search(features, k=1)
            results.append(result['top_1'])
            confidences.append(result['top_1_conf'])
        
        # Voting
        from collections import Counter
        vote_counts = Counter(results)
        best_drug = vote_counts.most_common(1)[0][0]
        avg_confidence = np.mean(confidences)
        
        return {
            'drug_name': best_drug,
            'confidence': avg_confidence,
            'votes': dict(vote_counts),
            'method': 'TTA'
        }


class LiveInferencePipeline:
    """Complete real-time inference pipeline"""
    
    def __init__(self,
                 camera_source: int = 0,
                 yolo_model: str = 'best_process_2.onnx',
                 index_path: str = 'faiss_database/drug_index.faiss',
                 metadata_path: str = 'faiss_database/metadata.json'):
        """
        Initialize pipeline
        
        Args:
            camera_source: Camera source
            yolo_model: YOLO model path
            index_path: FAISS index path
            metadata_path: Metadata JSON path
        """
        # Components
        self.frame_capture = AsyncFrameCapture(camera_source)
        self.drug_detector = YOLODrugDetector(yolo_model)
        self.preprocessor = DrugPreprocessor()
        self.feature_extractor = BatchFeatureExtractor()
        self.searcher = FAISSSearcher(index_path, metadata_path)
        self.tta = TestTimeAugmentation(self.feature_extractor, self.searcher)
        
        logger.info("✅ Initialized live inference pipeline")
    
    def run(self):
        """Run real-time inference"""
        logger.info("🚀 Starting live inference...")
        
        while True:
            frame = self.frame_capture.get_frame()
            
            if frame is None:
                continue
            
            # Detect drugs
            detections = self.drug_detector.detect(frame)
            
            # Process each detection
            results = []
            
            for detection in detections:
                crop = detection['crop']
                bbox = detection['bbox']
                
                # Preprocess
                preprocessed = self.preprocessor.preprocess(crop)
                tensor = self.preprocessor.normalize(preprocessed)
                
                # Extract features
                features = self.feature_extractor.extract_batch([tensor])
                
                # Search
                search_result = self.searcher.search(features[0], k=5)
                
                # Make decision
                decision = ConfidenceDecisionMaker.decide(search_result)
                
                # TTA if needed
                if decision['status'] == 'PARTIAL_MATCH':
                    tta_result = self.tta.augment_and_search(preprocessed)
                    if tta_result['confidence'] > 0.85:
                        decision['status'] = 'ACCEPT'
                        decision['top_1'] = tta_result['drug_name']
                        decision['confidence'] = tta_result['confidence']
                
                results.append({
                    'name': decision['top_1'],
                    'confidence': decision['confidence'],
                    'bbox': bbox,
                    'status': decision['status'],
                    'top_candidates': decision['top_3']
                })
            
            # Visualize
            self._visualize(frame, results)
            
            # Display
            cv2.imshow('PillTrack - Real-time Drug Identification', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def _visualize(self, frame: np.ndarray, results: List[Dict]):
        """Draw results on frame"""
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            name = result['name']
            conf = result['confidence']
            status = result['status']
            
            # Color based on status
            colors = {
                'ACCEPT': (0, 255, 0),
                'PARTIAL_MATCH': (0, 165, 255),
                'UNKNOWN': (0, 0, 255)
            }
            color = colors.get(status, (200, 200, 200))
            
            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{name} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Status indicator
            status_color = (0, 255, 0) if status == 'ACCEPT' else (0, 0, 255)
            cv2.putText(frame, status, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # FPS counter
        fps = self.frame_capture.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def cleanup(self):
        """Cleanup resources"""
        self.frame_capture.stop()
        cv2.destroyAllWindows()
        logger.info("✅ Pipeline closed")


if __name__ == '__main__':
    pipeline = LiveInferencePipeline(
        camera_source=0,
        yolo_model='best_process_2.onnx',
        index_path='faiss_database/drug_index.faiss',
        metadata_path='faiss_database/metadata.json'
    )
    
    pipeline.run()
