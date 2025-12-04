#!/usr/bin/env python3
"""
🔍 Debug YOLO Detection Issues
Test YOLO model directly to see what's wrong with detection output
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

def test_yolo_detection():
    """Test YOLO detection on a sample image"""
    
    print("\n" + "="*70)
    print("🔍 YOLO DETECTION DEBUG")
    print("="*70)
    
    # Step 1: Check model file
    print("\n📋 Step 1: Checking model file...")
    model_path = 'best_process_2.onnx'
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024*1024)
        print(f"✅ Model found: {model_path} ({size_mb:.1f} MB)")
    else:
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Step 2: Import ultralytics
    print("\n📦 Step 2: Importing ultralytics...")
    try:
        from ultralytics import YOLO
        print("✅ ultralytics imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ultralytics: {e}")
        print("   Install: pip install ultralytics")
        return False
    
    # Step 3: Load model
    print("\n🔧 Step 3: Loading YOLO model...")
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Task: {model.task}")
        print(f"   Model names: {model.names}")
        print(f"   Input size: {model.model.yaml.get('nc', '?')} classes")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Step 4: Create dummy image
    print("\n🖼️  Step 4: Creating test image...")
    
    # Create a simple test image (640x480, 3 channels)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"✅ Created test image: {test_image.shape}")
    
    # Save test image for reference
    cv2.imwrite('debug_test_image.png', test_image)
    print("   Saved to: debug_test_image.png")
    
    # Step 5: Run detection
    print("\n🎯 Step 5: Running detection on test image...")
    try:
        results = model(test_image, verbose=False, conf=0.5)
        print(f"✅ Detection completed")
        
        if results:
            print(f"\nDetection results:")
            print(f"  Number of result objects: {len(results)}")
            print(f"  Result type: {type(results[0])}")
            
            if hasattr(results[0], 'boxes'):
                boxes = results[0].boxes
                print(f"  Number of boxes: {len(boxes)}")
                
                if len(boxes) > 0:
                    print(f"\n  First 5 detections:")
                    for i, box in enumerate(boxes[:5]):
                        print(f"    {i}: conf={box.conf}, xyxy={box.xyxy}, shape={box.xyxy.shape}")
                else:
                    print(f"  ⚠️  No boxes detected (this is expected for random noise)")
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Test with real camera frame if available
    print("\n📷 Step 6: Testing with camera frame (if available)...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print(f"✅ Got camera frame: {frame.shape}")
                
                # Resize to 640x480
                frame_resized = cv2.resize(frame, (640, 480))
                
                print(f"🎯 Running detection on camera frame...")
                results = model(frame_resized, verbose=False, conf=0.6)
                
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    print(f"✅ Found {len(boxes)} detections")
                    
                    if len(boxes) > 0:
                        print(f"\n   Detailed detections:")
                        for i, box in enumerate(boxes[:10]):  # Show first 10
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            print(f"   {i}: bbox=({x1},{y1},{x2},{y2}) conf={conf:.3f} class={cls}")
                    else:
                        print("   No detections found (crop may not contain drugs)")
            else:
                print("⚠️  Could not read camera frame")
        else:
            print("⚠️  Camera not available")
    except Exception as e:
        print(f"⚠️  Camera test failed: {e}")
    
    # Step 7: Model info
    print("\n📊 Step 7: Model information...")
    try:
        print(f"  Model arch: {model.model.__class__.__name__}")
        print(f"  Model device: {model.device}")
        print(f"  Model type: {model.model_type}")
        
        # Try to get layer info
        if hasattr(model.model, 'model'):
            print(f"  Has internal model: Yes")
        else:
            print(f"  Has internal model: No")
    except Exception as e:
        print(f"⚠️  Could not get model info: {e}")
    
    print("\n" + "="*70)
    print("✅ DEBUG COMPLETE")
    print("="*70)
    print("\nIf you see too many detections (>50):")
    print("  - The model may be detecting noise as drugs")
    print("  - Increase confidence threshold in phase2_live_inference_pi5.py")
    print("  - Current: conf=0.6")
    print("  - Try: conf=0.7 or conf=0.75")
    print("\nIf you see no detections:")
    print("  - Model may be working correctly (no drugs in image)")
    print("  - Or model needs to be fine-tuned on your drug images")
    print("\n" + "="*70 + "\n")
    
    return True

if __name__ == '__main__':
    try:
        test_yolo_detection()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
