"""
Phase 1: Database Preparation Pipeline
========================================

This module handles:
1. Background removal using seg_db_best.pt
2. Image augmentation for 1-shot learning
3. Feature extraction using EfficientNet-B3
4. FAISS index construction
5. Metadata storage
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_DIM = 1536
IMAGE_SIZE = 300
BATCH_SIZE = 32

class BackgroundRemover:
    """Remove backgrounds from drug images using YOLO segmentation"""
    
    def __init__(self, model_path: str = 'seg_db_best.pt'):
        """
        Initialize background remover
        
        Args:
            model_path: Path to seg_db_best.pt segmentation model
        """
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        logger.info(f"✅ Loaded segmentation model from {model_path}")
    
    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Remove background and return isolated drug
        
        Args:
            image: Input image (H×W×3)
            
        Returns:
            Clean drug image with white background
        """
        # Run YOLO segmentation
        results = self.model(image)
        
        if results[0].masks is None:
            logger.warning("No mask detected, returning original image")
            return image
        
        # Get mask
        mask = results[0].masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Apply mask: white background
        h, w = image.shape[:2]
        background = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = np.where(mask_3ch[:,:,:3] > 0, image, background)
        
        return result
    
    def batch_remove_backgrounds(self, image_paths: List[str]) -> List[np.ndarray]:
        """Remove backgrounds from multiple images"""
        cleaned_images = []
        
        for path in tqdm(image_paths, desc="Removing backgrounds"):
            image = cv2.imread(str(path))
            if image is not None:
                cleaned = self.remove_background(image)
                cleaned_images.append(cleaned)
            else:
                logger.warning(f"Failed to load {path}")
        
        return cleaned_images


class AugmentationStrategy:
    """1-shot learning augmentation: Create 10 versions from 1 image"""
    
    def __init__(self, num_versions: int = 10):
        """
        Args:
            num_versions: Number of augmented versions per image
        """
        self.num_versions = num_versions
        self.augmenters = self._create_augmenters()
    
    def _create_augmenters(self) -> List:
        """Create list of augmentation pipelines"""
        augmenters = [
            # Version 0: Original (no augmentation)
            A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
            ]),
            
            # Versions 1-3: Rotation variations
            A.Compose([
                A.Rotate(limit=5, p=1.0, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
            ]),
            A.Compose([
                A.Rotate(limit=(-10, -5), p=1.0, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
            ]),
            A.Compose([
                A.Rotate(limit=(5, 10), p=1.0, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
            ]),
            
            # Versions 4-5: Scale variations
            A.Compose([
                A.Resize(height=int(IMAGE_SIZE * 1.05), 
                        width=int(IMAGE_SIZE * 1.05)),
                A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
            ]),
            A.Compose([
                A.Resize(height=int(IMAGE_SIZE * 0.95), 
                        width=int(IMAGE_SIZE * 0.95)),
                A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE,
                            border_mode=cv2.BORDER_CONSTANT, value=255),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
            ]),
            
            # Versions 6-8: Photometric variations
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, 
                                          contrast_limit=0, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
            ]),
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0), 
                                          contrast_limit=0, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
            ]),
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0, 
                                          contrast_limit=0.1, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
            ]),
            
            # Version 9: Combined augmentation
            A.Compose([
                A.Rotate(limit=5, p=0.7, border_mode=cv2.BORDER_CONSTANT),
                A.RandomBrightnessContrast(brightness_limit=0.05, 
                                          contrast_limit=0.05, p=0.7),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
            ]),
        ]
        
        return augmenters
    
    def augment(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply all augmentations to single image
        
        Args:
            image: Input image (H×W×3, uint8)
            
        Returns:
            List of 10 augmented versions
        """
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Resize to IMAGE_SIZE
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Apply augmentations
        augmented_versions = []
        for augmenter in self.augmenters:
            try:
                aug_image = augmenter(image=image)['image']
                augmented_versions.append(aug_image)
            except Exception as e:
                logger.error(f"Augmentation failed: {e}, using original")
                augmented_versions.append(image)
        
        return augmented_versions


class FeatureExtractor:
    """Extract 1536-dim features using EfficientNet-B3"""
    
    def __init__(self, model_name: str = 'efficientnet_b3', 
                 device: str = DEVICE):
        """
        Initialize feature extractor
        
        Args:
            model_name: Timm model name
            device: torch device
        """
        self.device = device
        
        # Load pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
            global_pool=''   # No pooling
        )
        self.backbone = self.backbone.to(device)
        self.backbone.eval()
        
        logger.info(f"✅ Loaded {model_name} feature extractor")
    
    def extract(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract features from image batch
        
        Args:
            image_tensor: (B, 3, 300, 300) normalized image tensor
            
        Returns:
            (B, 1536) feature vectors
        """
        with torch.no_grad():
            # Forward pass
            features = self.backbone(image_tensor)
            
            # Handle different output shapes
            if len(features.shape) == 4:
                # (B, C, H, W) → (B, C)
                features = F.adaptive_avg_pool2d(features, 1)
                features = features.squeeze(-1).squeeze(-1)
            
            # Ensure 1536-dim output
            if features.shape[-1] != FEATURE_DIM:
                # Project if needed
                logger.warning(f"Feature dim {features.shape[-1]} != {FEATURE_DIM}")
                # Use only first 1536 dims or pad
                if features.shape[-1] > FEATURE_DIM:
                    features = features[:, :FEATURE_DIM]
                else:
                    pad_size = FEATURE_DIM - features.shape[-1]
                    features = F.pad(features, (0, pad_size))
            
            # L2 normalize
            features = F.normalize(features, p=2, dim=1)
        
        return features.cpu()


class MultiscaleFeatureGenerator:
    """Generate multi-scale features for partial view robustness"""
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.extractor = feature_extractor
    
    def generate(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate features for full, 80%, and 60% crops
        
        Args:
            image: Input image (300×300)
            
        Returns:
            {
                'full': (1536,) features,
                'crop_80': (1536,) features,
                'crop_60': (1536,) features
            }
        """
        h, w = image.shape[:2]
        
        # Full image
        tensor_full = self._to_tensor(image)
        feat_full = self.extractor.extract(tensor_full).squeeze()
        
        # 80% center crop
        crop_80 = self._center_crop(image, 0.8)
        tensor_80 = self._to_tensor(crop_80)
        feat_80 = self.extractor.extract(tensor_80).squeeze()
        
        # 60% center crop
        crop_60 = self._center_crop(image, 0.6)
        tensor_60 = self._to_tensor(crop_60)
        feat_60 = self.extractor.extract(tensor_60).squeeze()
        
        return {
            'full': feat_full.numpy(),
            'crop_80': feat_80.numpy(),
            'crop_60': feat_60.numpy()
        }
    
    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        """Convert image to normalized tensor"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image / 255.0  # [0, 1]
        image = (image - np.array([0.485, 0.456, 0.406])) / \
                np.array([0.229, 0.224, 0.225])
        
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.float().to(DEVICE)
    
    @staticmethod
    def _center_crop(image: np.ndarray, scale: float) -> np.ndarray:
        """Center crop image"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        
        return image[start_y:start_y+new_h, start_x:start_x+new_w]


class FAISIndexBuilder:
    """Construct FAISS index for fast similarity search"""
    
    def __init__(self, dimension: int = FEATURE_DIM, 
                 use_gpu: bool = True):
        """
        Args:
            dimension: Feature dimension (1536)
            use_gpu: Use GPU for faster search
        """
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.index = None
        self.metadata = None
    
    def build(self, features: np.ndarray, 
              drug_names: List[str]) -> Tuple:
        """
        Build FAISS index from features
        
        Args:
            features: (N, 1536) feature vectors, L2-normalized
            drug_names: List of N drug names
            
        Returns:
            (index, metadata)
        """
        # Ensure features are float32
        features = features.astype('float32')
        
        # Create IndexFlatIP (Inner Product = Cosine for L2-norm)
        index = faiss.IndexFlatIP(self.dimension)
        
        # Add features
        index.add(features)
        
        # Move to GPU if available
        if self.use_gpu and torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("✅ FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Could not move to GPU: {e}")
        
        # Store metadata
        metadata = {
            'drug_names': drug_names,
            'num_drugs': len(drug_names),
            'feature_dim': self.dimension,
            'created': datetime.now().isoformat()
        }
        
        self.index = index
        self.metadata = metadata
        
        logger.info(f"✅ Built FAISS index with {len(drug_names)} drugs")
        return index, metadata
    
    def save(self, index_path: str, metadata_path: str):
        """Save index and metadata to disk"""
        if self.index is None:
            raise ValueError("No index built yet")
        
        # Save index
        faiss.write_index(faiss.index_gpu_to_cpu(self.index) 
                         if self.use_gpu else self.index, 
                         index_path)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"✅ Saved index to {index_path}")
        logger.info(f"✅ Saved metadata to {metadata_path}")


class MetadataDatabase:
    """SQLite database for storing drug metadata"""
    
    def __init__(self, db_path: str = 'drug_metadata.db'):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drugs (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                feature_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS multiscale_features (
                id INTEGER PRIMARY KEY,
                drug_id INTEGER NOT NULL,
                scale TEXT NOT NULL,
                feature_index INTEGER NOT NULL,
                FOREIGN KEY (drug_id) REFERENCES drugs(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Initialized metadata database at {self.db_path}")
    
    def insert_drug(self, name: str, feature_id: int):
        """Insert drug metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT OR REPLACE INTO drugs (name, feature_id) VALUES (?, ?)',
            (name, feature_id)
        )
        
        conn.commit()
        conn.close()


class DatabasePreparationPipeline:
    """Complete database preparation pipeline"""
    
    def __init__(self, 
                 data_folder: str = 'drug-scraping-c',
                 seg_model_path: str = 'seg_db_best.pt',
                 output_folder: str = 'faiss_database'):
        """
        Args:
            data_folder: Folder containing drug images
            seg_model_path: Path to segmentation model
            output_folder: Output folder for index and metadata
        """
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Initialize components
        self.bg_remover = BackgroundRemover(seg_model_path)
        self.augmenter = AugmentationStrategy(num_versions=10)
        self.feature_extractor = FeatureExtractor()
        self.multiscale_gen = MultiscaleFeatureGenerator(self.feature_extractor)
        self.index_builder = FAISIndexBuilder()
        self.metadata_db = MetadataDatabase()
    
    def run(self):
        """Execute complete pipeline"""
        logger.info("🚀 Starting database preparation pipeline...")
        
        # Step 1: Load and clean drug images
        drug_images, drug_names = self._load_and_clean_drugs()
        logger.info(f"✅ Loaded {len(drug_images)} drugs")
        
        # Step 2: Augment images
        augmented_images = self._augment_images(drug_images, drug_names)
        logger.info(f"✅ Created {len(augmented_images)} augmented images")
        
        # Step 3: Extract features
        all_features = self._extract_features(augmented_images)
        logger.info(f"✅ Extracted features from {len(all_features)} images")
        
        # Step 4: Build FAISS index
        self._build_index(all_features, drug_names)
        logger.info("✅ Built FAISS index")
        
        # Step 5: Generate multi-scale features
        self._generate_multiscale_features(drug_images, drug_names)
        logger.info("✅ Generated multi-scale features")
        
        logger.info("🎉 Database preparation complete!")
    
    def _load_and_clean_drugs(self) -> Tuple[List, List]:
        """Load drug images and remove backgrounds"""
        drug_images = []
        drug_names = []
        
        for drug_folder in sorted(self.data_folder.iterdir()):
            if not drug_folder.is_dir():
                continue
            
            # Get first image (1-shot learning)
            images = list(drug_folder.glob('*.jpg')) + \
                    list(drug_folder.glob('*.png'))
            
            if images:
                image = cv2.imread(str(images[0]))
                if image is not None:
                    cleaned = self.bg_remover.remove_background(image)
                    drug_images.append(cleaned)
                    drug_names.append(drug_folder.name)
        
        return drug_images, drug_names
    
    def _augment_images(self, images: List, names: List) -> Dict:
        """Augment images for 1-shot learning"""
        augmented = {}
        
        for image, name in tqdm(zip(images, names), desc="Augmenting"):
            augmented[name] = self.augmenter.augment(image)
        
        return augmented
    
    def _extract_features(self, augmented_images: Dict) -> np.ndarray:
        """Extract features from all augmented images"""
        all_features = []
        
        for drug_name, versions in tqdm(augmented_images.items(), 
                                       desc="Extracting features"):
            for version in versions:
                tensor = torch.from_numpy(version).unsqueeze(0).to(DEVICE)
                features = self.feature_extractor.extract(tensor)
                all_features.append(features.squeeze().numpy())
        
        return np.array(all_features)
    
    def _build_index(self, features: np.ndarray, drug_names: List):
        """Build and save FAISS index"""
        index, metadata = self.index_builder.build(features, drug_names)
        
        # Save
        self.index_builder.save(
            str(self.output_folder / 'drug_index.faiss'),
            str(self.output_folder / 'metadata.json')
        )
    
    def _generate_multiscale_features(self, images: List, names: List):
        """Generate and store multi-scale features"""
        multiscale_features = {}
        
        for image, name in tqdm(zip(images, names), desc="Multi-scale features"):
            multiscale_features[name] = self.multiscale_gen.generate(image)
        
        # Save
        with open(self.output_folder / 'multiscale_features.pkl', 'wb') as f:
            pickle.dump(multiscale_features, f)


if __name__ == '__main__':
    # Run complete pipeline
    pipeline = DatabasePreparationPipeline(
        data_folder='drug-scraping-c',
        seg_model_path='seg_db_best.pt',
        output_folder='faiss_database'
    )
    
    pipeline.run()
