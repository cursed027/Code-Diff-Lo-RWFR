import os
import glob
import sys
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch.nn.functional as F
from torchvision import transforms

# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_ROOT  = os.path.join(PROJECT_ROOT, "results")   # results/<lora_xxxx>/*.png
GT_DIR       = os.path.join(PROJECT_ROOT, "gt")        # gt/*.png
ADAFACE_DIR  = os.path.join(PROJECT_ROOT, "AdaFace")

# =========================================================
# Safety: AdaFace import
# =========================================================
if not os.path.isdir(ADAFACE_DIR):
    raise RuntimeError(
        "AdaFace not found. Clone it inside the repo:\n"
        "git clone https://github.com/mk-minchul/AdaFace.git"
    )

sys.path.insert(0, ADAFACE_DIR)
from AdaFace.net import build_model

# =========================================================
# IQA metrics (NTIRE-consistent)
# =========================================================
from pyiqa import create_metric

device = "cuda"

clip_iqa = create_metric("clipiqa", device=device)
maniqa   = create_metric("maniqa", device=device)
musiq    = create_metric("musiq", device=device)
niqe     = create_metric("niqe", device=device)

# (Optional – NOT required for checkpoint selection)
# qalign = create_metric("qalign", device=device)

# =========================================================
# AdaFace model
# =========================================================
adaface = build_model("ir_50").to(device).eval()

tf_112 = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def load_img(path):
    return Image.open(path).convert("RGB")

@torch.no_grad()
def adaface_sim(img1, img2):
    x1 = tf_112(img1).unsqueeze(0).to(device)
    x2 = tf_112(img2).unsqueeze(0).to(device)
    f1, _ = adaface(x1)
    f2, _ = adaface(x2)
    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)
    return (f1 * f2).sum().item()

# =========================================================
# NTIRE-style evaluation
# =========================================================
IDENTITY_THRESHOLD = 0.80   # ✔ aligned with winners table

results = []

for ckpt_dir in sorted(os.listdir(RESULT_ROOT)):
    out_dir = os.path.join(RESULT_ROOT, ckpt_dir)
    if not os.path.isdir(out_dir):
        continue

    sims, clips, manis, mus, niqs = [], [], [], [], []

    img_list = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    if len(img_list) == 0:
        continue

    print(f"\n▶ Evaluating {ckpt_dir} ({len(img_list)} images)")

    for img_path in tqdm(img_list, desc=ckpt_dir):
        name = os.path.basename(img_path)
        gt_path = os.path.join(GT_DIR, name)
        if not os.path.exists(gt_path):
            continue

        out_img = load_img(img_path)
        gt_img  = load_img(gt_path)

        sims.append(adaface_sim(out_img, gt_img))
        clips.append(clip_iqa(img_path).item())
        manis.append(maniqa(img_path).item())
        mus.append(musiq(img_path).item())
        niqs.append(niqe(img_path).item())

    mean_id = float(np.mean(sims))

    # -------------------------------
    # Identity gate (NTIRE behavior)
    # -------------------------------
    if mean_id < IDENTITY_THRESHOLD:
        print(f"[REJECT] {ckpt_dir} | AdaFace={mean_id:.3f}")
        continue

    # -------------------------------
    # NTIRE weighted score
    # -------------------------------
    score = (
        np.mean(clips)
        + np.mean(manis)
        + np.mean(mus) / 100.0
        + max(0.0, (10.0 - np.mean(niqs)) / 10.0)
    )

    results.append({
        "checkpoint": ckpt_dir,
        "score": score,
        "adaface": mean_id,
        "clipiqa": np.mean(clips),
        "maniqa": np.mean(manis),
        "musiq": np.mean(mus),
        "niqe": np.mean(niqs),
    })

# =========================================================
# Final ranking
# =========================================================
results = sorted(results, key=lambda x: x["score"], reverse=True)

print("\n================ FINAL RANKING ================")
for r in results:
    print(
        f"{r['checkpoint']} | "
        f"Score={r['score']:.4f} | "
        f"ID={r['adaface']:.3f}"
    )

if len(results) > 0:
    print("\n🏆 BEST CHECKPOINT:", results[0]["checkpoint"])
else:
    print("\n❌ No checkpoint passed identity gate")
