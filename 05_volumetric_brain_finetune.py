"""
05_volumetric_brain_finetune.py
=================================
Patient-level transfer learning on whole-brain studies (NIfTI / DICOM / slices).
Adapts Swin_5C, ConvNext_5C, MONAI_5C to real patient volumetric data.

Skip  weights exist + Volumetric_Finetune.json fingerprint unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, cast

import cv2
import nibabel as nib       # type: ignore[import-untyped]
import numpy as np
import pydicom
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets.swin_unetr import SwinUNETR
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms  # type: ignore[import-untyped]
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

DIAGNOSTIC_LABELS: list[str] = [
    "Glioma", "Meningioma", "No Tumor",
    "Pituitary", "Tumor (Generic / CT)",
]
BRANCH_WEIGHTS: list[float] = [0.4, 0.3, 0.3]
NO_TUMOR_IDX: int           = 2
NUM_CLASSES: int            = len(DIAGNOSTIC_LABELS)


# ── MONAI adapter ─────────────────────────────────────────────────────────────

class MedicalSwinAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = SwinUNETR(
            spatial_dims=2, in_channels=3, out_channels=14, feature_size=24,
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(384, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feats: torch.Tensor = self.backbone.swinViT(x, normalize=True)[-1]
        return self.classifier(feats)


def _remap(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("swin.swinViT."):
            out[k.replace("swin.swinViT.", "backbone.swinViT.")] = v
        elif k.startswith("swin."):
            out[k.replace("swin.", "backbone.")] = v
        else:
            out[k] = v
    return out


def _load_monai(model: nn.Module, path: Path, device: torch.device) -> None:
    sd: Dict[str, torch.Tensor] = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(_remap(sd), strict=False)


def _fingerprint(root: str | Path, exts: Iterable[str] | None = None) -> str:
    rp = Path(root)
    if not rp.exists(): return ""
    allowed: set[str] | None = ({e.lower() for e in exts} if exts else None)
    d = hashlib.sha256()
    for p in sorted(f for f in rp.rglob("*") if f.is_file()):
        if allowed and not any(str(p).lower().endswith(e) for e in allowed):
            continue
        st = p.stat()
        d.update(str(p.relative_to(rp)).encode())
        d.update(str(st.st_size).encode())
        d.update(str(int(st.st_mtime)).encode())
    return d.hexdigest()


# ── Config ────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP    = DEVICE.type == "cuda"

VOL_DIR    = Path("dataset_volumetric")
SENTINEL   = Path("Volumetric_Finetune.json")      
W_SWIN     = Path("Swin_5C.pth")
W_CONV     = Path("ConvNext_5C.pth")
W_MONAI    = Path("MONAI_5C.pth")
REQUIRED_W = [W_SWIN, W_CONV, W_MONAI]

MAX_SLICES = 240
MIN_SLICES = 8
TRIM       = 0.15
INPUT_SZ   = 256
BATCH_SZ   = 12
FT_EPOCHS  = 4
LR         = 1e-4
WD         = 1e-4

IMG_EXT = (".jpg", ".jpeg", ".png")
VOL_EXT = (".nii", ".nii.gz")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatientStudy:
    patient_id:  str
    label:       int
    source_kind: str
    source_path: str
    slice_count: int


@dataclass(frozen=True)
class SliceRecord:
    patient_id:  str
    label:       int
    source_kind: str
    source_path: str
    slice_index: int | None = None


# ── Utilities ─────────────────────────────────────────────────────────────────

def _norm(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    return np.uint8(np.clip((a - lo) / (hi - lo + 1e-8) * 255, 0, 255))


def _uniform(length: int, cap: int) -> list[int]:
    if length <= 0: return []
    if length <= cap: return list(range(length))
    return cast(list[int], np.linspace(0, length - 1, num=cap, dtype=int).tolist())


def _nifti_indices(path: Path, cap: int) -> list[int]:
    img = nib.load(str(path))   # type: ignore[reportPrivateImportUsage]
    vol = img.get_fdata()       # type: ignore[attr-defined]
    depth = int(vol.shape[2]) if hasattr(vol, "shape") and len(vol.shape) >= 3 else 0  # type: ignore[union-attr]
    start = int(depth * TRIM); end = int(depth * (1 - TRIM))
    return [start + i for i in _uniform(max(end - start, 0), cap)]


def _sorted_dcm(folder: Path) -> list[Path]:
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"]
    order: list[tuple[float, Path]] = []
    for p in paths:
        try:
            h = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)  # type: ignore[union-attr]
            key: float = (
                float(h.ImagePositionPatient[2]) if hasattr(h, "ImagePositionPatient")  # type: ignore[union-attr]
                else float(h.InstanceNumber) if hasattr(h, "InstanceNumber")             # type: ignore[union-attr]
                else float(len(order))
            )
        except Exception:
            key = float(len(order))
        order.append((key, p))
    order.sort(key=lambda x: x[0])
    return [p for _, p in order]


def _sorted_imgs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


# ── Patient discovery ─────────────────────────────────────────────────────────

def _discover(root: Path) -> list[PatientStudy]:
    studies: list[PatientStudy] = []
    class_map = {"tumor": 1, "Tumor": 1, "no_tumor": 0, "NoTumor": 0}
    for cls_name, label in class_map.items():
        cls_dir = root / cls_name
        if not cls_dir.is_dir(): continue
        for entry in sorted(cls_dir.iterdir()):
            pid = f"{cls_name}/{entry.stem}"
            nl = entry.name.lower()
            if entry.is_file() and (nl.endswith(".nii.gz") or nl.endswith(".nii")):
                idxs = _nifti_indices(entry, MAX_SLICES)
                if len(idxs) >= MIN_SLICES:
                    studies.append(PatientStudy(pid, label, "nifti", str(entry), len(idxs)))
                continue
            if not entry.is_dir(): continue
            dcms = _sorted_dcm(entry)
            if dcms:
                sel = _uniform(len(dcms), MAX_SLICES)
                if len(sel) >= MIN_SLICES:
                    studies.append(PatientStudy(pid, label, "dicom_series", str(entry), len(sel)))
                continue
            imgs = _sorted_imgs(entry)
            sel = _uniform(len(imgs), MAX_SLICES)
            if len(sel) >= MIN_SLICES:
                studies.append(PatientStudy(pid, label, "image_series", str(entry), len(sel)))
    return studies


# ── Dataset ───────────────────────────────────────────────────────────────────

class VolumetricSliceDataset(Dataset[tuple[Any, int, str]]):
    def __init__(self, studies: list[PatientStudy], transform: Any = None) -> None:
        self.transform = transform
        self.samples: list[SliceRecord] = []
        self._nifti_cache: dict[str, np.ndarray] = {}
        for s in studies:
            self.samples.extend(self._index(s))

    def _index(self, s: PatientStudy) -> list[SliceRecord]:
        p = Path(s.source_path); recs: list[SliceRecord] = []
        if s.source_kind == "nifti":
            for i in _nifti_indices(p, MAX_SLICES):
                recs.append(SliceRecord(s.patient_id, s.label, "nifti", str(p), i))
        elif s.source_kind == "dicom_series":
            files = _sorted_dcm(p)
            for i in _uniform(len(files), MAX_SLICES):
                recs.append(SliceRecord(s.patient_id, s.label, "dicom_file", str(files[i])))
        elif s.source_kind == "image_series":
            files = _sorted_imgs(p)
            for i in _uniform(len(files), MAX_SLICES):
                recs.append(SliceRecord(s.patient_id, s.label, "image_file", str(files[i])))
        return recs

    def _load(self, r: SliceRecord) -> Image.Image:
        if r.source_kind == "nifti":
            if r.source_path not in self._nifti_cache:
                io = nib.load(r.source_path)   # type: ignore[reportPrivateImportUsage]
                self._nifti_cache[r.source_path] = io.get_fdata()  # type: ignore[attr-defined]
            vol: np.ndarray = self._nifti_cache[r.source_path]
            raw = _norm(vol[:, :, r.slice_index or 0])
            return Image.fromarray(np.stack([raw, raw, raw], axis=2))
        if r.source_kind == "dicom_file":
            ds = pydicom.dcmread(r.source_path)  # type: ignore[union-attr]
            arr = _norm(ds.pixel_array.astype(np.float32))  # type: ignore[union-attr]
            if arr.ndim == 2: arr = np.stack([arr, arr, arr], axis=2)
            return Image.fromarray(arr)
        img_cv = cv2.imread(r.source_path)
        if img_cv is None: raise IOError(f"Cannot read {r.source_path}")
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int, str]:
        r = self.samples[idx]
        try:
            img = self._load(r)
        except Exception:
            img = Image.new("RGB", (INPUT_SZ, INPUT_SZ), 0)
        if self.transform is not None:
            img = self.transform(img)
        return img, r.label, r.patient_id


# ── Skip logic ────────────────────────────────────────────────────────────────

def _skip_check(fp: str) -> bool:
    ok = all(p.exists() and p.stat().st_size > 5*1024*1024 for p in REQUIRED_W)
    if not ok or not SENTINEL.exists(): return False
    try:
        meta: Dict[str, Any] = json.loads(SENTINEL.read_text())
    except json.JSONDecodeError:
        return False
    if meta.get("dataset_fingerprint") == fp:
        print(f"[SKIP] Volumetric fine-tune done ({meta.get('timestamp','?')}) — fingerprint unchanged.")
        return True
    print("[INFO] Dataset changed — re-running fine-tune.")
    return False


# ── Freeze strategy ───────────────────────────────────────────────────────────

def _freeze(model: nn.Module, kind: str) -> None:
    for p in model.parameters(): p.requires_grad = False
    if kind == "swin":
        for p in model.head.parameters(): p.requires_grad = True  # type: ignore[union-attr]
        if hasattr(model, "norm"):
            for p in model.norm.parameters(): p.requires_grad = True  # type: ignore[union-attr]
    elif kind == "conv":
        for p in model.head.parameters(): p.requires_grad = True  # type: ignore[union-attr]
        if hasattr(model, "norm_pre"):
            for p in model.norm_pre.parameters(): p.requires_grad = True  # type: ignore[union-attr]
    elif kind == "monai":
        for p in model.classifier.parameters(): p.requires_grad = True  # type: ignore[union-attr]
        layers = getattr(getattr(model, "backbone", None), "swinViT", None)
        if layers and hasattr(layers, "layers"):
            for p in layers.layers[-1].parameters(): p.requires_grad = True


def _load_council() -> tuple[nn.Module, nn.Module, nn.Module]:
    missing = [str(p) for p in REQUIRED_W if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing weights (run 04 first): {missing}")
    swin: nn.Module = timm.create_model("swinv2_tiny_window8_256", pretrained=False, num_classes=5).to(DEVICE)
    conv: nn.Module = timm.create_model("convnextv2_nano",          pretrained=False, num_classes=5).to(DEVICE)
    monai: nn.Module = MedicalSwinAdapter().to(DEVICE)
    swin.load_state_dict(torch.load(W_SWIN,  map_location=DEVICE, weights_only=True))
    conv.load_state_dict(torch.load(W_CONV,  map_location=DEVICE, weights_only=True))
    _load_monai(monai, W_MONAI, DEVICE)
    _freeze(swin, "swin"); _freeze(conv, "conv"); _freeze(monai, "monai")
    print(f"  [OK] SwinV2   ← {W_SWIN}")
    print(f"  [OK] ConvNeXt ← {W_CONV}")
    print(f"  [OK] MONAI    ← {W_MONAI}")
    return swin, conv, monai


def _tumor_logit(logits: torch.Tensor) -> torch.Tensor:
    tp = 1.0 - torch.softmax(logits.float(), 1)[:, NO_TUMOR_IDX]
    return torch.logit(torch.clamp(tp, 1e-6, 1 - 1e-6))


def _evaluate(
    swin: nn.Module, conv: nn.Module, monai: nn.Module,
    loader: DataLoader[Any],
) -> dict[str, float]:
    st: List[int] = []; sp: List[int] = []
    pt: dict[str, int] = {}; ps: dict[str, list[np.ndarray]] = {}
    for m in (swin, conv, monai): m.eval()
    with torch.no_grad():
        for imgs, lbls, pids in loader:
            imgs = imgs.to(DEVICE)
            ll: List[int] = cast(List[int], lbls.tolist())
            with torch.amp.autocast("cuda", enabled=AMP):  # type: ignore[attr-defined]
                ps_t = torch.softmax(swin(imgs), 1)
                pc_t = torch.softmax(conv(imgs), 1)
            pm_t = torch.softmax(monai(imgs.float()), 1)
            cons = BRANCH_WEIGHTS[0]*ps_t.float() + BRANCH_WEIGHTS[1]*pc_t.float() + BRANCH_WEIGHTS[2]*pm_t.float()
            sp.extend(cast(List[int], (cons[:, NO_TUMOR_IDX] < 0.5).long().cpu().tolist()))
            st.extend(ll)
            for i, pid in enumerate(pids):
                pid_s = str(pid); pt[pid_s] = ll[i]
                ps.setdefault(pid_s, []).append(cons[i].cpu().numpy())
    pl: List[int] = []; pp: List[int] = []
    for pid_s, scores in ps.items():
        avg: np.ndarray = np.mean(scores, axis=0)
        pl.append(pt[pid_s]); pp.append(int(avg[NO_TUMOR_IDX] < 0.5))
    return {
        "slice_accuracy":   round(float(accuracy_score(cast(Any, st), cast(Any, sp))), 4),
        "slice_macro_f1":   round(float(f1_score(cast(Any, st), cast(Any, sp), average="macro")), 4),
        "patient_accuracy": round(float(accuracy_score(cast(Any, pl), cast(Any, pp))), 4),
        "patient_macro_f1": round(float(f1_score(cast(Any, pl), cast(Any, pp), average="macro")), 4),
    }


def _finetune(
    swin: nn.Module, conv: nn.Module, monai: nn.Module,
    tr_ld: DataLoader[Any], va_ld: DataLoader[Any],
) -> dict[str, float]:
    crit   = nn.BCEWithLogitsLoss()
    params = [p for m in (swin, conv, monai) for p in m.parameters() if p.requires_grad]
    opt    = optim.AdamW(params, lr=LR, weight_decay=WD)
    sch    = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS)
    scaler = torch.amp.GradScaler("cuda") if AMP else None  # type: ignore[attr-defined]

    for ep in range(FT_EPOCHS):
        for m in (swin, conv, monai): m.train()
        ep_loss = 0.0
        bar = tqdm(tr_ld, desc=f"[FT ep{ep+1}/{FT_EPOCHS}]")
        for imgs, lbls, _ in bar:
            imgs, lbls_f = imgs.to(DEVICE), lbls.float().to(DEVICE)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):  # type: ignore[attr-defined]
                    loss = (crit(_tumor_logit(swin(imgs)), lbls_f) +
                            crit(_tumor_logit(conv(imgs)), lbls_f) +
                            crit(_tumor_logit(monai(imgs.float())), lbls_f))
                scaler.scale(loss).backward()   # type: ignore[union-attr]
                scaler.step(opt)                # type: ignore[union-attr]
                scaler.update()                 # type: ignore[union-attr]
            else:
                loss = (crit(_tumor_logit(swin(imgs)), lbls_f) +
                        crit(_tumor_logit(conv(imgs)), lbls_f) +
                        crit(_tumor_logit(monai(imgs.float())), lbls_f))
                loss.backward(); opt.step()
            ep_loss += float(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4f}")  # type: ignore[union-attr]
        sch.step()  # type: ignore[union-attr]
        print(f"  ep{ep+1} mean loss: {ep_loss/max(len(tr_ld),1):.4f}")

    metrics = _evaluate(swin, conv, monai, va_ld)
    print(f"\n  Slice Acc {metrics['slice_accuracy']*100:.2f}%  F1 {metrics['slice_macro_f1']:.4f}")
    print(f"  Patient Acc {metrics['patient_accuracy']*100:.2f}%  F1 {metrics['patient_macro_f1']:.4f}")
    return metrics


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_volumetric_finetune() -> None:
    print("=" * 70)
    print("  Brain Tumor Detection — Volumetric Transfer Learning")
    print("=" * 70)

    if not VOL_DIR.is_dir():
        print(f"[CRITICAL] '{VOL_DIR}' not found. Run 01b first."); return

    fp = _fingerprint(VOL_DIR, IMG_EXT + VOL_EXT + (".dcm",))
    if not fp:
        print("[CRITICAL] No volumetric files found."); return
    if _skip_check(fp): return

    studies = _discover(VOL_DIR)
    print(f"[INFO] Discovered {len(studies)} patient studies.")
    if len(studies) < 4:
        print("[CRITICAL] Need ≥ 4 patient studies (2 per class)."); return

    labels_list: list[int] = [s.label for s in studies]
    split = train_test_split(studies, test_size=0.2, stratify=labels_list, random_state=42)
    tr_st: list[PatientStudy] = cast(list[PatientStudy], split[0])
    va_st: list[PatientStudy] = cast(list[PatientStudy], split[1])

    tfm_tr = transforms.Compose([
        transforms.Resize((INPUT_SZ, INPUT_SZ)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8), transforms.ColorJitter(.12, .12),
        transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3),
    ])
    tfm_va = transforms.Compose([
        transforms.Resize((INPUT_SZ, INPUT_SZ)),
        transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3),
    ])

    tr_ds = VolumetricSliceDataset(tr_st, tfm_tr)
    va_ds = VolumetricSliceDataset(va_st, tfm_va)
    if len(tr_ds) == 0 or len(va_ds) == 0:
        print("[CRITICAL] Dataset empty after indexing."); return

    kw: dict[str, Any] = {"num_workers": 2, "pin_memory": AMP}
    tr_ld: DataLoader[Any] = DataLoader(tr_ds, batch_size=BATCH_SZ, shuffle=True,  **kw)
    va_ld: DataLoader[Any] = DataLoader(va_ds, batch_size=BATCH_SZ, shuffle=False, **kw)

    print(f"  Train: {len(tr_st)} patients / {len(tr_ds)} slices")
    print(f"  Val:   {len(va_st)} patients / {len(va_ds)} slices | Device: {DEVICE}")

    try:
        swin, conv, monai = _load_council()
    except FileNotFoundError as e:
        print(f"[CRITICAL] {e}"); return

    print(f"\n[INFO] Fine-tuning {FT_EPOCHS} epochs …\n")
    metrics = _finetune(swin, conv, monai, tr_ld, va_ld)

    torch.save(swin.state_dict(), W_SWIN)
    torch.save(conv.state_dict(), W_CONV)
    torch.save(monai.state_dict(), W_MONAI)

    meta: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "dataset_fingerprint": fp,
        "patient_count": len(studies),
        "slice_count": len(tr_ds) + len(va_ds),
        "epochs": FT_EPOCHS, "batch_size": BATCH_SZ, "lr": LR,
        "device": str(DEVICE), **metrics,
    }
    SENTINEL.write_text(json.dumps(meta, indent=2))
    print(f"\n[OK]  {W_SWIN}\n[OK]  {W_CONV}\n[OK]  {W_MONAI}\n[OK]  {SENTINEL}")
    print("=" * 70)


if __name__ == "__main__":
    run_volumetric_finetune()
