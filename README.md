

# Code-Diff-Lo-RWFR

**Robust Wild Face Restoration with Diffusion, LoRA, and CodeFormer Guidance**

This repository implements a **Robust Wild Face Restoration (RWFR)** pipeline based on **DiffBIR v2**, enhanced with:

* **LoRA fine-tuning** (UNet + ControlNet),
* **on-the-fly CodeFormer guidance** for face priors,
* support for **unaligned real-world faces** (NTIRE-style setting).

The project supports:

* **Stage-2 fine-tuning** (LoRA training),
* **Inference on unaligned images** with CodeFormer-guided Diffusion.

## 📌 Overview of the Pipeline

**Inference flow (Unaligned BFR):**

1. Input image (unaligned, wild face)
2. Face detection & alignment (DiffBIR internal)
3. **CodeFormer restores aligned face (512×512)**
4. CodeFormer output is injected into ControlNet
5. Diffusion restores face + background
6. Faces are pasted back into the original image

👉 CodeFormer is used **only as a face prior**, not as a standalone enhancer.

---

## 📁 Repository Structure (Key Files)

```
Code-Diff-Lo-RWFR/
├── diffbir/
│   ├── inference/
│   │   ├── loop.py
│   │   └── unaligned_bfr_loop.py      # main unaligned inference logic
│   ├── utils/
│   │   ├── common.py                  # model loading (ckpt + safetensors)
│   │   └── codeformer_wrapper.py      # CodeFormer face restorer
│   └── model/
│       └── cldm.py
├── configs/
│   ├── train/
│   │   └── train_stage2.yaml          # LoRA training config
│   └── inference/
│       ├── bsrnet.yaml
│       └── swinir.yaml
├── train_stage2.py
├── inference.py
├── requirements.txt
└── README.md
```

---

## 🔽 Required Pretrained Weights

### 1️⃣ Stage-1 Face Restoration Weights

```bash
wget https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt
```

---

### 2️⃣ Stable Diffusion v2.1 Base (SafeTensors)

```bash
wget https://huggingface.co/Manojb/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors
```

#### ✅ Convert SafeTensors → `.ckpt`

```python
from safetensors.torch import load_file
import torch

ckpt = load_file("v2-1_512-ema-pruned.safetensors")
torch.save(
    {"state_dict": ckpt, "global_step": 0},
    "v2-1_512-ema-pruned.ckpt"
)

print("✅ Converted safetensors → ckpt")
sd = torch.load("v2-1_512-ema-pruned.ckpt", map_location="cpu", weights_only=False)
print(type(sd), sd.keys())
```

**Expected output**

```
✅ Converted safetensors → ckpt
<class 'dict'> dict_keys(['state_dict', 'global_step'])
```

---

## 🧪 Training: Stage-2 (LoRA Fine-Tuning)

### Prerequisites

* Python ≥ 3.9
* CUDA GPU recommended
* Make sure **LQ, GT, and CodeFormer outputs share identical filenames**

---

### Step-by-Step Training Guide

1️⃣ Clone the repository

```bash
git clone <this-repo-url>
cd Code-Diff-Lo-RWFR
```

2️⃣ Install dependencies
Install `requirements.txt`.
Torch & xFormers **must be installed manually** depending on GPU.

> ⚠️ On **Pascal GPUs (P100)**:
>
> * Do **not** install xFormers
> * Use system torch

> ⚡ On **Ampere / RTX GPUs**:

```bash
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.27.post2 --no-deps
```

3️⃣ Prepare dataset

* Place **LQ / GT / CF** images
* Filenames must match exactly

4️⃣ Configure training
Edit:

```
configs/train/train_stage2.yaml
```

Key sections to adjust:

* `train:` (batch size, lr, iterations)
* `dataset:` (paths)
* `lora:` (rank_unet, rank_controlnet)

5️⃣ Launch training

```bash
accelerate launch train_stage2.py \
  --config configs/train/train_stage2.yaml
```

LoRA checkpoints will be saved in the experiment directory.

---

## 🚀 Inference: Unaligned Face Restoration

### Important Note (CodeFormer Dependency)

This project **requires CodeFormer** during inference.

CodeFormer is **not vendored** into this repo by design.

---

### Step-by-Step Inference Guide

#### 1️⃣ Clone both repositories

```bash
git clone <this-repo-url>
git clone https://github.com/sczhou/CodeFormer
```

Directory layout:

```
.
├── Code-Diff-Lo-RWFR
└── CodeFormer
```
---

## 🔐 AdaFace (Identity Loss Dependency)

This project optionally uses **AdaFace** as an **identity embedding network**
during **CHECKPOINT SELECTION only**.

AdaFace is **NOT vendored** into this repository by design.

---

### 📥 Install AdaFace (One-Time)

Clone the official AdaFace repository **inside in this repo**:

```bash
cd dir/Code-Diff-Lo-RWFR
git clone https://github.com/mk-minchul/AdaFace.git
```

Install Dependencies **(SKIP ALREADY INCLUDED IN MAIN REQ.TXT OF THIS REPO!!)**


⚠️ Do NOT install AdaFace’s original PyTorch or Lightning requirements.
This project uses modern PyTorch (2.x) and only requires AdaFace
for forward embedding inference, not training.
---

#### 2️⃣ Install dependencies

```bash
pip install -r Code-Diff-Lo-RWFR/requirements.txt
```

Install Torch / xFormers **as per your GPU** (see training section).

---

#### 3️⃣ Initialize BasicSR (mandatory)

```bash
cd CodeFormer
python basicsr/setup.py develop
```

⚠️ This step is required for CodeFormer to register its architectures.

---

#### 4️⃣ Prepare inputs

* LQ images (unaligned faces)
* LoRA weights from training
* Stage-1 + SD v2.1 weights (CodeFormer weights download automatically)

---

#### 5️⃣ Run inference

```bash
cd Code-Diff-Lo-RWFR

python inference.py \
  --task unaligned_face \
  --upscale 2 \
  --version v2 \
  --sampler spaced \
  --steps 50 \
  --cfg_scale 4.0 \
  --captioner none \
  --pos_prompt "" \
  --neg_prompt "low quality, blurry, low-resolution, noisy, unsharp, weird textures" \
  --input /path/to/LQ_images \
  --output /path/to/output_dir \
  --lora_path /path/to/lora_checkpoint.pt \
  --rank_unet 64 \
  --rank_controlnet 16 \
  --batch_size 1 \
  --n_samples 1 \
  --precision fp32 \
  --device cuda
```

### 🏆 Checkpoint Selection (NTIRE-Style)

After running inference for multiple LoRA checkpoints:

```bash
    python diffbir/inference/select_best_lora.py
```

The script:

- Computes CLIP-IQA, MANIQA, MUSIQ, NIQE

- Applies AdaFace identity gating

- Ranks checkpoints using NTIRE weighted score

Output:

    🏆 BEST CHECKPOINT: lora_0000200

---

## 🔧 Where to Modify Things

| Task                               | File                                      |
| ---------------------------------- | ----------------------------------------- |
| Training config                    | `configs/train/train_stage2.yaml`         |
| Inference logic                    | `diffbir/inference/unaligned_bfr_loop.py` |
| CodeFormer usage                   | `diffbir/utils/codeformer_wrapper.py`     |
| Model loading (ckpt / safetensors) | `diffbir/utils/common.py`                 |
| Main inference CLI                 | `inference.py`                            |

---

## 🧠 Key Design Choices (For NTIRE Reviewers)

* **CodeFormer is used only on aligned face crops**
* Background fusion is handled by DiffBIR
* LoRA is trained jointly on UNet + ControlNet
* ControlNet guidance is preserved via CF face priors
* No double face detection or alignment occurs

This design avoids identity drift and boundary artifacts while improving realism.

---

## 📜 License & Credits

* DiffBIR: original authors
* CodeFormer: Zhou et al.
* Stable Diffusion v2.1: Stability AI
* NTIRE RWFR Challenge

---

