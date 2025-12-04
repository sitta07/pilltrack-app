"""
Phase 2: Live Inference Pipeline - Raspberry Pi 5 Optimized
============================================================

This module handles real-time drug identification from camera feed.

Optimizations for Raspberry Pi 5:
- Picamera2 instead of OpenCV (native Pi 5 support)
- Reduced batch size (1 vs 32)
- CPU-only inference
- Single detection per frame (no multi-batch)
- Memory-efficient processing
- Frame skipping for consistent timing
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from threading import Thread, Lock, Event
from queue import Queue, Empty
import time
import logging
from pathlib import Path
import json
import pickle
import faiss
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pi 5 Configuration
IS_RASPBERRY_PI = True
DEVICE = torch.device('cpu')
IMAGE_SIZE = 300
BATCH_SIZE = 1
FPS_TARGET = 15  # Reduced FPS target for Pi (vs 30 for GPU)

# ⚡ PERFORMANCE OPTIMIZATION FOR PI5
SKIP_DISPLAY = True  # Skip cv2.imshow (huge speedup!)
SKIP_FRAMES = 1  # Process every Nth frame (1=all, 2=every 2nd, etc)
PROCESS_RESOLUTION = (320, 240)  # Lower resolution for faster processing
DETECTION_CONFIDENCE_THRESHOLD = 0.3  # Skip crops below this
ENABLE_FRAME_SKIPPING = True  # Drop frames if can't keep up

logger.info(f"🔧 Running on: {DEVICE} (Raspberry Pi 5 Mode)")
logger.info(f"⚙️  Target FPS: {FPS_TARGET}")
logger.info(f"⚡ OPTIMIZATION: SKIP_DISPLAY={SKIP_DISPLAY}, SKIP_FRAMES={SKIP_FRAMES}")

# Try to import Picamera2 for Pi, fallback to OpenCV
try:
    from picamera2 import Picamera2
    CAMERA_TYPE = 'picamera2'
    logger.info("📷 Using Picamera2 (native Pi support)")
except ImportError:
    CAMERA_TYPE = 'opencv'
    logger.warning("⚠️  Picamera2 not available, using OpenCV camera")


@dataclass
class DetectionResult:
    """Detection result container"""
    drug_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    crop: np.ndarray
    tier: str  # 'TIER1', 'TIER2', 'MANUAL'


class AsyncFrameCapture:
    """Capture frames from camera with buffering (thread-safe)"""
    
    def __init__(self, camera_source: int = 0, buffer_size: int = 1):
        """
        Initialize camera capture
        
        Args:
            camera_source: Camera device ID or path
            buffer_size: Frame buffer size (⚡ reduced to 1 for lower latency)
        """
        self.camera_source = camera_source
        self.buffer_size = buffer_size
        self.frame_queue = Queue(maxsize=buffer_size)
        self.running = False
        self.capture_thread = None
        self.latest_frame = None
        self.lock = Lock()
    
    def start(self):
        """Start camera capture thread"""
        if CAMERA_TYPE == 'picamera2':
            self._start_picamera2()
        else:
            self._start_opencv()
    
    def _start_picamera2(self):
        """Start Picamera2 capture"""
        try:
            self.cap = Picamera2(self.camera_source)
            # Configure for 480p (lighter load on Pi)
            self.cap.configure(self.cap.create_preview_configuration(main={"size": (640, 480)}))
            self.cap.start()
            logger.info("✅ Picamera2 started (640x480)")
        except Exception as e:
            logger.error(f"❌ Picamera2 failed: {e}, falling back to OpenCV")
            self._start_opencv()
    
    def _start_opencv(self):
        """Start OpenCV camera capture"""
        self.cap = cv2.VideoCapture(self.camera_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
        logger.info("✅ OpenCV camera started (640x480)")
    
    def _capture_loop(self):
        """Capture frames in background thread"""
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            try:
                if CAMERA_TYPE == 'picamera2':
                    frame = self.cap.capture_array()
                else:
                    ret, frame = self.cap.read()
                    if not ret:
                        continue
                
                frame_count += 1
                
                # Calculate FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = 30 / elapsed
                    logger.debug(f"📷 Capture FPS: {fps:.1f}")
                    start_time = time.time()
                
                # Put in queue (discard old frames if buffer full)
                try:
                    self.frame_queue.put_nowait(frame)
                except:
                    # Queue full, try to remove oldest
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
                
                with self.lock:
                    self.latest_frame = frame
                
            except Exception as e:
                logger.warning(f"⚠️  Capture error: {e}")
                time.sleep(0.1)
    
    def start_capture_thread(self):
        """Start the capture thread"""
        self.running = True
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("🔄 Frame capture thread started")
    
    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get latest frame from queue"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            # Return latest frame if queue empty
            with self.lock:
                return self.latest_frame
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        
        if hasattr(self, 'cap'):
            if CAMERA_TYPE == 'picamera2':
                self.cap.stop()
            else:
                self.cap.release()
        logger.info("🛑 Camera stopped")


class YOLODrugDetector:
    """Detect drugs in frame using YOLO"""
    
    def __init__(self, model_path: str = 'best_process_2.onnx'):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model (ONNX or PT)
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"✅ Loaded YOLO model: {model_path}")
        except ImportError:
            logger.error("❌ ultralytics not installed")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect drugs in frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of detections {bbox, crop, conf}
        """
        try:
            # Ensure frame is 3-channel (BGR)
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                if len(frame.shape) == 4:  # Remove batch dimension if present
                    frame = frame[0]
                if frame.shape[2] == 4:  # Convert BGRA to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                elif frame.shape[2] == 1:  # Convert grayscale to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            results = self.model(frame, verbose=False, conf=0.5)
            detections = []
            
            if results and results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    
                    # Extract crop
                    crop = frame[y1:y2, x1:x2]
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'crop': crop,
                        'conf': conf
                    })
            
            return detections
        except Exception as e:
            logger.warning(f"⚠️  Detection failed: {e}")
            return []


class DrugPreprocessor:
    """Preprocess drug crop for feature extraction"""
    
    def __init__(self, target_size: int = 300):
        self.target_size = target_size
    
    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Resize and pad to target size with white background
        
        Args:
            crop: Input crop (BGR)
            
        Returns:
            Preprocessed image (300x300, RGB normalized)
        """
        # Resize maintaining aspect ratio
        h, w = crop.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(crop, (new_w, new_h))
        
        # Pad to target size with white
        top = (self.target_size - new_h) // 2
        bottom = self.target_size - new_h - top
        left = (self.target_size - new_w) // 2
        right = self.target_size - new_w - left
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        # Normalize to [0, 1] and convert BGR->RGB (make contiguous copy)
        normalized = padded.astype(np.float32) / 255.0
        normalized = np.ascontiguousarray(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
        
        return normalized


class BatchFeatureExtractor:
    """Extract features from drug crops"""
    
    def __init__(self, model_name: str = 'efficientnet_b3'):
        """
        Initialize feature extractor
        
        Args:
            model_name: Name of timm model
        """
        try:
            import timm
            self.model = timm.create_model(model_name, pretrained=True, features_only=False)
            self.model = self.model.to(DEVICE)
            self.model.eval()
            logger.info(f"✅ Loaded feature extractor: {model_name}")
        except Exception as e:
            logger.error(f"❌ Cannot load model: {e}")
            raise
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from batch of images
        
        Args:
            images: List of images (300x300, RGB normalized)
            
        Returns:
            Feature vectors (1536-dim, L2-normalized)
        """
        if not images:
            return np.array([])
        
        # Convert to tensor
        tensors = []
        for img in images:
            tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            tensors.append(tensor)
        
        batch = torch.stack(tensors).to(DEVICE)
        
        with torch.no_grad():
            features = self.model.forward_features(batch)
            features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
            features = F.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()


class FAISSSearcher:
    """Search FAISS index for nearest neighbors"""
    
    def __init__(self, index_path: str, metadata_path: str, k: int = 5):
        """
        Initialize searcher
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata JSON
            k: Number of nearest neighbors
        """
        # Check if database files exist
        if not os.path.exists(index_path):
            logger.error(f"❌ Database not found: {index_path}")
            logger.error("   Run phase1_database_preparation_pi5.py first")
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        if not os.path.exists(metadata_path):
            logger.error(f"❌ Metadata not found: {metadata_path}")
            logger.error("   Run phase1_database_preparation_pi5.py first")
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            self.drug_names = data['drug_names']
        
        # Check if database is empty
        if len(self.drug_names) == 0:
            logger.error("❌ Database is empty! No drugs loaded.")
            logger.error("   Run phase1_database_preparation_pi5.py again to populate the database")
            raise ValueError("Database is empty - no drugs loaded from metadata")
        
        self.k = k
        logger.info(f"✅ Loaded FAISS index ({len(self.drug_names)} drugs)")
    
    def search(self, features: np.ndarray, k: Optional[int] = None) -> Dict:
        """
        Search for nearest neighbors
        
        Args:
            features: Query feature vector
            k: Number of neighbors (default: self.k)
            
        Returns:
            Dictionary with top matches
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        k = k or self.k
        distances, indices = self.index.search(features.astype(np.float32), k)
        
        results = {
            'distances': distances[0],
            'indices': indices[0],
            'top_1': self.drug_names[indices[0][0]],
            'top_1_conf': float(distances[0][0]),
            'top_3': [self.drug_names[idx] for idx in indices[0][:3]]
        }
        
        return results


class ConfidenceDecisionMaker:
    """Make confidence-based decisions using 3-tier system"""
    
    def __init__(self, accept_threshold: float = 0.75, partial_threshold: float = 0.60):
        """
        Initialize decision maker
        
        Args:
            accept_threshold: Confidence threshold for acceptance
            partial_threshold: Confidence threshold for partial match
        """
        self.accept_threshold = accept_threshold
        self.partial_threshold = partial_threshold
    
    def decide(self, confidence: float) -> str:
        """
        Make decision based on confidence score
        
        Args:
            confidence: Confidence score [0, 1]
            
        Returns:
            Decision: 'ACCEPT', 'PARTIAL', or 'UNKNOWN'
        """
        if confidence >= self.accept_threshold:
            return 'ACCEPT'
        elif confidence >= self.partial_threshold:
            return 'PARTIAL'
        else:
            return 'UNKNOWN'


class LiveInferencePipeline:
    """End-to-end real-time inference pipeline"""
    
    def __init__(self,
                 camera_source: int = 0,
                 yolo_model: str = 'best_process_2.onnx',
                 index_path: str = 'faiss_database/drug_index.faiss',
                 metadata_path: str = 'faiss_database/metadata.json'):
        """
        Initialize pipeline
        
        Args:
            camera_source: Camera device ID
            yolo_model: Path to YOLO model
            index_path: Path to FAISS index
            metadata_path: Path to metadata
        """
        logger.info("🚀 Initializing Live Inference Pipeline (Pi 5 Mode)")
        
        # Initialize components
        self.camera = AsyncFrameCapture(camera_source)
        self.detector = YOLODrugDetector(yolo_model)
        self.preprocessor = DrugPreprocessor()
        self.extractor = BatchFeatureExtractor()
        self.searcher = FAISSSearcher(index_path, metadata_path)
        self.decision_maker = ConfidenceDecisionMaker()
        
        logger.info("✅ Pipeline initialized")
    
    def run(self):
        """Run main inference loop - OPTIMIZED FOR PI5"""
        logger.info("🎬 Starting inference loop (press 'q' to quit)...")
        logger.info(f"📊 Display: {'OFF (faster)' if SKIP_DISPLAY else 'ON'}")
        logger.info(f"⚡ Processing resolution: {PROCESS_RESOLUTION}")
        
        self.camera.start()
        self.camera.start_capture_thread()
        
        frame_count = 0
        processed_count = 0
        detection_count = 0
        inference_times = []
        last_fps_log = time.time()
        
        try:
            while True:
                frame_start = time.time()
                
                # Get frame
                frame = self.camera.get_frame(timeout=0.5)
                if frame is None:
                    continue
                
                frame_count += 1
                
                # ⚡ Skip frames for speed
                if frame_count % SKIP_FRAMES != 0:
                    continue
                
                processed_count += 1
                
                # ⚡ Resize for faster processing
                small_frame = cv2.resize(frame, PROCESS_RESOLUTION)
                
                # Detect drugs
                detections = self.detector.detect(small_frame)
                
                # ⚡ Optimization: Scale bboxes back to original resolution
                scale_x = frame.shape[1] / PROCESS_RESOLUTION[0]
                scale_y = frame.shape[0] / PROCESS_RESOLUTION[1]
                
                # Process detections
                for det in detections:
                    try:
                        # ⚡ Skip low-confidence crops
                        if det['conf'] < DETECTION_CONFIDENCE_THRESHOLD:
                            continue
                        
                        crop = det['crop']
                        
                        # ⚡ Quick validation
                        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                            continue
                        
                        # Preprocess
                        preprocessed = self.preprocessor.preprocess(crop)
                        
                        # Extract features
                        features = self.extractor.extract_batch([preprocessed])
                        
                        # Search
                        result = self.searcher.search(features[0])
                        
                        # Decide
                        confidence = result['top_1_conf']
                        decision = self.decision_maker.decide(confidence)
                        
                        detection_count += 1
                        
                        # ⚡ Only draw if DISPLAY enabled
                        if not SKIP_DISPLAY:
                            # Scale bbox back
                            x1, y1, x2, y2 = det['bbox']
                            x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                            
                            color = self._get_color(decision)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label
                            label = f"{result['top_1'][:20]} ({confidence:.2f}) [{decision}]"
                            cv2.putText(frame, label, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    except Exception as e:
                        logger.debug(f"⚠️  Processing error: {e}")
                
                # Calculate timing
                frame_time = time.time() - frame_start
                inference_times.append(frame_time)
                if len(inference_times) > 30:
                    inference_times.pop(0)
                
                # ⚡ Only display if enabled
                if not SKIP_DISPLAY:
                    avg_time = np.mean(inference_times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    cv2.imshow('PillTrack - Drug Identification', frame)
                    
                    # Check for quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("⏹️  Stopping inference...")
                        break
                else:
                    # ⚡ Check for quit without display
                    if processed_count % 60 == 0:
                        key_event = cv2.waitKey(1) & 0xFF
                        if key_event == ord('q'):
                            logger.info("⏹️  Stopping inference...")
                            break
                
                # Log progress periodically
                if time.time() - last_fps_log > 5:  # Log every 5 seconds
                    avg_time = np.mean(inference_times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    logger.info(f"✨ Processed {processed_count} frames, {detection_count} detections, FPS: {fps:.1f}")
                    last_fps_log = time.time()
        
        except KeyboardInterrupt:
            logger.info("⏹️  Interrupted by user")
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            logger.info("✅ Inference stopped")
    
    def _get_color(self, decision: str) -> Tuple[int, int, int]:
        """Get color for decision"""
        colors = {
            'ACCEPT': (0, 255, 0),      # Green
            'PARTIAL': (0, 255, 255),    # Yellow
            'UNKNOWN': (0, 0, 255)       # Red
        }
        return colors.get(decision, (255, 255, 255))


if __name__ == '__main__':
    pipeline = LiveInferencePipeline(
        camera_source=0,
        yolo_model='best_process_2.onnx',
        index_path='faiss_database/drug_index.faiss',
        metadata_path='faiss_database/metadata.json'
    )
    pipeline.run()
