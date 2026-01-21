import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
import timm
from monai.networks.nets import SwinUNETR
from tqdm import tqdm
import os
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 PHASE 4: Training Council (Resumable + 5-Class F1)...")

class MonaiSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = SwinUNETR(spatial_dims=2, in_channels=3, out_channels=14, feature_size=24)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(384, 5))
    def forward(self, x): return self.head(self.swin.swinViT(x, normalize=True)[-1])

# --- HELPER: LOAD PRETRAINED IF EXISTS ---
def load_or_create_models():
    print("   ...Checking for existing checkpoints...")
    
    # Create Architecture
    swin = timm.create_model('swinv2_tiny_window8_256', pretrained=True, num_classes=5)
    conv = timm.create_model('convnextv2_nano', pretrained=True, num_classes=5)
    monai = MonaiSwin()
    
    # Load Weights if available
    if os.path.exists("HYDRA_Swin_5C.pth"):
        swin.load_state_dict(torch.load("HYDRA_Swin_5C.pth", map_location=DEVICE))
        print("      ✅ Resumed SwinV2")
        
    if os.path.exists("HYDRA_ConvNext_5C.pth"):
        conv.load_state_dict(torch.load("HYDRA_ConvNext_5C.pth", map_location=DEVICE))
        print("      ✅ Resumed ConvNeXt")
        
    if os.path.exists("HYDRA_MONAI_5C.pth"):
        monai.load_state_dict(torch.load("HYDRA_MONAI_5C.pth", map_location=DEVICE))
        print("      ✅ Resumed MONAI")
        
    return swin.to(DEVICE), conv.to(DEVICE), monai.to(DEVICE)

def train_council_gold():
    # 1. Data
    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    if not os.path.exists("dataset_council"): print("❌ Error: Dataset missing."); return
    full_ds = datasets.ImageFolder("dataset_council", transform=tfm)
    
    train_idx, val_idx = train_test_split(range(len(full_ds)), test_size=0.2, stratify=full_ds.targets)
    
    labels = [full_ds.targets[i] for i in train_idx]
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=16, num_workers=2, pin_memory=True)

    # 2. Models (Resumable)
    swin, conv, monai = load_or_create_models()
    
    opt = optim.AdamW(list(swin.parameters())+list(conv.parameters())+list(monai.parameters()), lr=1e-4)
    crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # 3. Train
    for ep in range(12):
        swin.train(); conv.train(); monai.train()
        loop = tqdm(train_loader, desc=f"Ep {ep+1}/12")
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(swin(x),y) + crit(conv(x),y) + crit(monai(x),y)
            loss.backward(); opt.step()
            loop.set_postfix(loss=loss.item())

    # 4. Metrics
    print("\n📊 Calculating Final Metrics...")
    swin.eval(); conv.eval(); monai.eval()
    all_preds, all_targs = [], []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            p1 = torch.softmax(swin(x), 1)
            p2 = torch.softmax(conv(x), 1)
            p3 = torch.softmax(monai(x), 1)
            avg = (p1 * 0.4) + (p2 * 0.3) + (p3 * 0.3)
            all_preds.extend(torch.argmax(avg, 1).cpu().numpy())
            all_targs.extend(y.cpu().numpy())

    report = classification_report(all_targs, all_preds, target_names=full_ds.classes, digits=4)
    f1 = f1_score(all_targs, all_preds, average='macro')
    
    print("\n" + "="*60 + "\n🏥 FINAL CLINICAL REPORT CARD\n" + "="*60)
    print(report)
    print(f"🏆 Overall Macro F1: {f1:.4f}")
    print("="*60)

    torch.save(swin.state_dict(), "HYDRA_Swin_5C.pth")
    torch.save(conv.state_dict(), "HYDRA_ConvNext_5C.pth")
    torch.save(monai.state_dict(), "HYDRA_MONAI_5C.pth")
    print("✅ Models Updated.")

if __name__ == "__main__":
    train_council_gold()