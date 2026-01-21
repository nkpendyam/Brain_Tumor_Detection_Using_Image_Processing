import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import shutil
import glob
import json
import random
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_gatekeeper_gold():
    print("🚀 PHASE 2: Training Gatekeeper (Randomized & Balanced)...")
    
    base = "dataset_gatekeeper_gold"
    if os.path.exists(base): shutil.rmtree(base)
    os.makedirs(f"{base}/Brain", exist_ok=True)
    os.makedirs(f"{base}/NotBrain", exist_ok=True)
    
    # 1. BRAINS (Randomized Selection)
    all_brains = glob.glob("dataset_council/*/*.jpg") + glob.glob("dataset_council/*/*.jpeg")
    random.shuffle(all_brains) # Shuffle to avoid bias
    selected_brains = all_brains[:3000] # Take 3000 random brains (MRI+CT)
    
    print(f"   -> Loading {len(selected_brains)} Brains...")
    for f in selected_brains: shutil.copy(f, f"{base}/Brain/")
    
    # 2. NEGATIVES
    xrays = glob.glob("dataset_negatives/*.jpg")[:1500]
    faces = glob.glob("dataset_faces/*.jpg")[:1500]
    for f in xrays + faces: shutil.copy(f, f"{base}/NotBrain/")
    
    # 3. TRAIN
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(base, transform=tfm)
    with open("gatekeeper_classes.json", "w") as f: json.dump(dataset.class_to_idx, f)
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = models.efficientnet_b0(weights='DEFAULT')
    model.classifier[1] = nn.Linear(1280, 2)
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    
    for ep in range(5):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"   Ep {ep+1} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "HYDRA_Gatekeeper_Clinical.pth")
    print("✅ Gatekeeper Armed.")

if __name__ == "__main__":
    train_gatekeeper_gold()