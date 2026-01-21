import os
import shutil
from ultralytics import YOLO

def setup_gold_data():
    print("🚀 PHASE 1: Clinical Data Setup (5-Class + Safety)...")
    
    home = os.path.expanduser("~")
    if not os.path.exists(os.path.join(home, ".kaggle", "kaggle.json")):
        print("❌ CRITICAL: kaggle.json missing."); return

    # A. COUNCIL DATA (5 Classes)
    base_dir = "dataset_council"
    if os.path.exists(base_dir): shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. MRI (Classes 1-4)
    print("   ⬇️  Fetching MRI Data...")
    os.system("kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p temp_mri --unzip")
    
    for root, _, files in os.walk("temp_mri"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                dest = None
                if "glioma" in root.lower(): dest = "Glioma"
                elif "meningioma" in root.lower(): dest = "Meningioma"
                elif "pituitary" in root.lower(): dest = "Pituitary"
                elif "no" in root.lower(): dest = "NoTumor"
                
                if dest:
                    os.makedirs(f"{base_dir}/{dest}", exist_ok=True)
                    shutil.copy(os.path.join(root, file), f"{base_dir}/{dest}/mri_{file}")

    # 2. CT (Class 5: Generic Tumor)
    print("   ⬇️  Fetching CT Data...")
    os.system("kaggle datasets download -d ahmedhamada0/brain-tumor-detection -p temp_ct --unzip")

    os.makedirs(f"{base_dir}/Tumor_Generic", exist_ok=True)
    
    for root, _, files in os.walk("temp_ct"):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                if "yes" in root.lower():
                    # CT Tumor -> Generic Class (Honest labeling)
                    shutil.copy(os.path.join(root, file), f"{base_dir}/Tumor_Generic/ct_{file}")
                elif "no" in root.lower():
                    # CT Healthy -> NoTumor
                    shutil.copy(os.path.join(root, file), f"{base_dir}/NoTumor/ct_{file}")

    shutil.rmtree("temp_mri"); shutil.rmtree("temp_ct")

    # B. GATEKEEPER DATA (Negatives)
    if not os.path.exists("dataset_negatives"):
        print("   ⬇️  Fetching Negatives (X-Ray/Faces)...")
        os.makedirs("dataset_negatives", exist_ok=True)
        os.makedirs("dataset_faces", exist_ok=True)
        
        os.system("kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p temp_xray --unzip")
        os.system("kaggle datasets download -d jessicali9530/celeba-dataset -p temp_faces --unzip")
        
        count = 0
        for root, _, files in os.walk("temp_xray"):
            for f in files:
                if f.endswith('.jpeg') and count < 1500:
                    shutil.copy(os.path.join(root, f), f"dataset_negatives/xray_{count}.jpg"); count+=1
        
        count = 0
        for root, _, files in os.walk("temp_faces"):
            for f in files:
                if f.endswith('.jpg') and count < 1500:
                    shutil.copy(os.path.join(root, f), f"dataset_faces/face_{count}.jpg"); count+=1
                    
        shutil.rmtree("temp_xray"); shutil.rmtree("temp_faces")

    print("✅ PHASE 1 COMPLETE: 5-Class Architecture Ready.")

if __name__ == "__main__":
    setup_gold_data()