"""
Phase 1: Database Preparation Pipeline - Raspberry Pi 5 Optimized
==================================================================

This module handles:
1. Background removal using seg_db_best.pt
2. Image augmentation for 1-shot learning
3. Feature extraction using EfficientNet-B3
4. FAISS index construction (CPU version for Pi)
5. Metadata storage

Optimizations for Raspberry Pi 5:
- CPU-only inference (no GPU dependency)
- Reduced batch size (8 vs 32)
- Lower model precision options
- Memory-efficient processing
- Threading for parallel operations
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from pathlib import Path
import pickle
import json
import faiss
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import sqlite3
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pi 5 Configuration - CPU optimized
IS_RASPBERRY_PI = True
DEVICE = torch.device('cpu')  # Force CPU on Pi
FEATURE_DIM = 1536
IMAGE_SIZE = 300
BATCH_SIZE = 8  # Reduced for Pi memory constraints
NUM_WORKERS = 2  # Light threading
ENABLE_HALF_PRECISION = False  # Disable for stability on Pi

logger.info(f"🔧 Running on: {DEVICE} (Raspberry Pi 5 Mode)")
logger.info(f"📊 Batch size: {BATCH_SIZE}")


class BackgroundRemover:
    """Remove backgrounds from drug images using YOLO segmentation"""
    
    def __init__(self, model_path: str = 'seg_db_best.pt'):
        """
        Initialize background remover
        
        Args:
            model_path: Path to segmentation model
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"✅ Loaded segmentation model: {model_path}")
        except ImportError:
            logger.error("❌ ultralytics not installed. Install: pip install ultralytics")
            raise
    
    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Remove background from single image
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Image with white background
        """
        try:
            results = self.model(image, verbose=False)
            mask = results[0].masks.data[0].cpu().numpy() if results[0].masks else None
            
            if mask is None:
                return image
            
            # Apply mask
            mask = (mask > 0.5).astype(np.uint8)
            mask_3d = np.stack([mask] * 3, axis=-1)
            output = image * mask_3d + 255 * (1 - mask_3d)
            return output.astype(np.uint8)
        except Exception as e:
            logger.warning(f"⚠️ Background removal failed: {e}, returning original image")
            return image
    
    def batch_remove_backgrounds(self, image_paths: List[str], max_workers: int = 2) -> List[np.ndarray]:
        """Remove backgrounds from multiple images with threading"""
        cleaned_images = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for path in image_paths:
                future = executor.submit(self._process_single, path)
                futures.append(future)
            
            for future in tqdm(futures, desc="🔄 Removing backgrounds", disable=False):
                try:
                    img = future.result()
                    if img is not None:
                        cleaned_images.append(img)
                except Exception as e:
                    logger.warning(f"⚠️ Error processing image: {e}")
        
        return cleaned_images
    
    def _process_single(self, image_path: str) -> Optional[np.ndarray]:
        """Process single image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"⚠️ Cannot read: {image_path}")
                return None
            return self.remove_background(img)
        except Exception as e:
            logger.warning(f"⚠️ Error reading {image_path}: {e}")
            return None


class AugmentationStrategy:
    """Create 10 augmented versions per image for 1-shot learning"""
    
    def __init__(self, num_versions: int = 10):
        self.num_versions = num_versions
    
    def augment(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Create multiple augmented versions of single image
        
        Args:
            image: Input image
            
        Returns:
            List of augmented images
        """
        augmentations = [
            A.Compose([A.NoOp()]),  # Original
            A.Compose([A.Rotate(limit=10, p=1.0)]),  # Rotation +10°
            A.Compose([A.Rotate(limit=-10, p=1.0)]),  # Rotation -10°
            A.Compose([A.Scale(scale=(0.95, 1.0), p=1.0)]),  # Zoom out
            A.Compose([A.Scale(scale=(1.0, 1.05), p=1.0)]),  # Zoom in
            A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.0, p=1.0)]),
            A.Compose([A.RandomBrightnessContrast(brightness_limit=-0.1, contrast_limit=0.0, p=1.0)]),
            A.Compose([A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.1, p=1.0)]),
            A.Compose([A.Rotate(limit=5, p=1.0), A.RandomBrightnessContrast(brightness_limit=0.05, p=1.0)]),
            A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)]),
        ]
        
        augmented = []
        for i, aug in enumerate(augmentations[:self.num_versions]):
            try:
                augmented_img = aug(image=image)['image']
                augmented.append(augmented_img)
            except Exception as e:
                logger.warning(f"⚠️ Augmentation {i} failed: {e}")
                augmented.append(image)
        
        return augmented


class FeatureExtractor:
    """Extract features using EfficientNet-B3"""
    
    def __init__(self, model_name: str = 'efficientnet_b3', use_half_precision: bool = False):
        """
        Initialize feature extractor
        
        Args:
            model_name: Name of timm model
            use_half_precision: Use FP16 for faster inference (Pi may have issues)
        """
        try:
            self.model = timm.create_model(model_name, pretrained=True, features_only=False)
            self.model = self.model.to(DEVICE)
            self.model.eval()
            self.use_half_precision = use_half_precision and (DEVICE.type == 'cuda')
            logger.info(f"✅ Loaded feature extractor: {model_name}")
        except Exception as e:
            logger.error(f"❌ Cannot load model: {e}")
            raise
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from single image
        
        Args:
            image: Input image (BGR, normalized)
            
        Returns:
            Feature vector (1536-dim, L2-normalized)
        """
        # Convert to tensor
        if isinstance(image, np.ndarray):
            # Assume image is already normalized [0, 1]
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        else:
            tensor = image.to(DEVICE)
        
        with torch.no_grad():
            features = self.model.forward_features(tensor)
            features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
            features = F.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()[0]
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract features from batch of images"""
        if not images:
            return np.array([])
        
        # Stack images
        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            else:
                tensor = img
            tensors.append(tensor)
        
        batch = torch.stack(tensors).to(DEVICE)
        
        with torch.no_grad():
            features = self.model.forward_features(batch)
            features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
            features = F.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()


class MultiscaleFeatureGenerator:
    """Generate multi-scale features for robust matching"""
    
    def __init__(self, extractor: FeatureExtractor):
        self.extractor = extractor
    
    def generate(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate features at multiple scales
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with 'full', '80', '60' scale features
        """
        h, w = image.shape[:2]
        
        # Full image
        full_features = self.extractor.extract(image)
        
        # 80% crop (center)
        h80 = int(h * 0.8)
        w80 = int(w * 0.8)
        y_start = (h - h80) // 2
        x_start = (w - w80) // 2
        crop_80 = image[y_start:y_start+h80, x_start:x_start+w80]
        crop_80 = cv2.resize(crop_80, (300, 300))
        features_80 = self.extractor.extract(crop_80)
        
        # 60% crop (center)
        h60 = int(h * 0.6)
        w60 = int(w * 0.6)
        y_start = (h - h60) // 2
        x_start = (w - w60) // 2
        crop_60 = image[y_start:y_start+h60, x_start:x_start+w60]
        crop_60 = cv2.resize(crop_60, (300, 300))
        features_60 = self.extractor.extract(crop_60)
        
        return {
            'full': full_features,
            '80': features_80,
            '60': features_60
        }


class FAISIndexBuilder:
    """Build FAISS index for similarity search"""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = {}
        self.drug_names = []
    
    def add(self, features: np.ndarray, drug_name: str, drug_id: int):
        """Add feature vector to index"""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        self.index.add(features.astype(np.float32))
        self.metadata[len(self.drug_names)] = {
            'drug_id': drug_id,
            'drug_name': drug_name
        }
        self.drug_names.append(drug_name)
    
    def save(self, index_path: str, metadata_path: str):
        """Save index and metadata"""
        os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
        
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'w') as f:
            json.dump({
                'drug_names': self.drug_names,
                'metadata': self.metadata,
                'dimension': self.dimension,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"✅ Index saved: {index_path}")
        logger.info(f"✅ Metadata saved: {metadata_path}")


class DatabasePreparationPipeline:
    """Orchestrate entire Phase 1 pipeline"""
    
    def __init__(self, 
                 data_folder: str = 'drug-scraping-c',
                 seg_model_path: str = 'seg_db_best.pt',
                 output_folder: str = 'faiss_database'):
        """
        Initialize pipeline
        
        Args:
            data_folder: Folder with drug images
            seg_model_path: Path to segmentation model
            output_folder: Output folder for FAISS index
        """
        self.data_folder = data_folder
        self.seg_model_path = seg_model_path
        self.output_folder = output_folder
        
        os.makedirs(output_folder, exist_ok=True)
        
        logger.info(f"📁 Input folder: {data_folder}")
        logger.info(f"📁 Output folder: {output_folder}")
    
    def run(self):
        """Run complete pipeline"""
        logger.info("🚀 Starting Phase 1: Database Preparation (Pi 5 Optimized)")
        
        try:
            # Step 1: Initialize components
            logger.info("\n📦 Initializing components...")
            remover = BackgroundRemover(self.seg_model_path)
            extractor = FeatureExtractor()
            multiscale_gen = MultiscaleFeatureGenerator(extractor)
            index_builder = FAISIndexBuilder()
            
            # Step 2: Load drug images
            logger.info(f"\n📂 Loading drug images from {self.data_folder}...")
            drug_folders = [d for d in os.listdir(self.data_folder) 
                          if os.path.isdir(os.path.join(self.data_folder, d))]
            drug_folders = sorted(drug_folders)[:100]  # Limit for Pi testing
            
            logger.info(f"📊 Found {len(drug_folders)} drug types")
            
            # Step 3: Process each drug
            multiscale_features = {}
            
            for drug_idx, drug_name in enumerate(tqdm(drug_folders, desc="🔄 Processing drugs")):
                drug_path = os.path.join(self.data_folder, drug_name)
                image_files = [f for f in os.listdir(drug_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if not image_files:
                    continue
                
                # Use first image
                img_path = os.path.join(drug_path, image_files[0])
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # Preprocess
                img_clean = remover.remove_background(img)
                img_resized = cv2.resize(img_clean, (IMAGE_SIZE, IMAGE_SIZE))
                img_normalized = img_resized.astype(np.float32) / 255.0
                img_normalized = img_normalized[..., ::-1]  # BGR to RGB
                
                # Extract multi-scale features
                scales = multiscale_gen.generate(img_normalized)
                
                # Average features for index
                avg_features = np.mean([scales['full'], scales['80'], scales['60']], axis=0)
                avg_features = avg_features / (np.linalg.norm(avg_features) + 1e-8)
                
                # Add to index
                index_builder.add(avg_features, drug_name, drug_idx)
                multiscale_features[drug_name] = scales
            
            # Step 4: Save index
            logger.info(f"\n💾 Saving FAISS index ({len(index_builder.drug_names)} drugs)...")
            index_path = os.path.join(self.output_folder, 'drug_index.faiss')
            metadata_path = os.path.join(self.output_folder, 'metadata.json')
            index_builder.save(index_path, metadata_path)
            
            # Step 5: Save multiscale features
            logger.info("💾 Saving multi-scale features...")
            features_path = os.path.join(self.output_folder, 'multiscale_features.pkl')
            with open(features_path, 'wb') as f:
                pickle.dump(multiscale_features, f)
            
            logger.info(f"\n✅ Phase 1 Complete!")
            logger.info(f"📊 Processed {len(index_builder.drug_names)} drugs")
            logger.info(f"📁 Index size: {os.path.getsize(index_path) / 1024 / 1024:.2f} MB")
            logger.info(f"🔧 Ready for Phase 2: Live Inference")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise


if __name__ == '__main__':
    pipeline = DatabasePreparationPipeline(
        data_folder='drug-scraping-c',
        seg_model_path='seg_db_best.pt',
        output_folder='faiss_database'
    )
    pipeline.run()
