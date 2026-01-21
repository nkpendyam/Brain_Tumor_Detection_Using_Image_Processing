from ultralytics import YOLO
import torch
import os

def train_hunter_gold():
    print("🚀 PHASE 3: Training Hunter (Resumable Turbo Mode)...")
    torch.cuda.empty_cache()
    
    # 1. CHECK FOR EXISTING WEIGHTS
    # If we trained before, load the best weights to fine-tune
    existing_weights = "runs/detect/hydra_hunter_clinical/weights/best.pt"
    
    if os.path.exists(existing_weights):
        print(f"   🔄 Found existing model at {existing_weights}. Resuming training...")
        model = YOLO(existing_weights)
    else:
        print("   ✨ No previous run found. Starting fresh...")
        model = YOLO("yolo11n.pt") 
    
    # 2. TRAIN
    results = model.train(
        data="brain-tumor.yaml",
        epochs=35,
        imgsz=640,
        
        # Stability Settings
        batch=8,
        workers=2,
        cache='disk',
        device=0,
        
        # Performance
        amp=True,
        augment=True,
        mosaic=0.5,
        mixup=0.0,
        
        optimizer="AdamW",
        lr0=0.001,
        
        name="hydra_hunter_clinical",
        exist_ok=True # Overwrite graph plots, but we loaded weights above
    )
    
    # 3. EXPORT
    if os.path.exists(existing_weights):
        trained_model = YOLO(existing_weights)
        export_path = trained_model.export(format="torchscript")
        print(f"✅ Hunter Trained & Exported to: {export_path}")

if __name__ == "__main__":
    train_hunter_gold()