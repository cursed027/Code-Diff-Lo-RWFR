import sys
import torch
import numpy as np
from torchvision.transforms.functional import normalize

# 🔑 Add CodeFormer repo to Python path
# CHANGE THIS PATH if your CodeFormer is elsewhere
CODEFORMER_ROOT = "/content/CodeFormer"
if CODEFORMER_ROOT not in sys.path:
    sys.path.insert(0, CODEFORMER_ROOT)

# --- CodeFormer imports (official, unmodified) ---
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY


class CodeFormerWrapper:
    """
    CodeFormer used as an aligned-face restorer ONLY.
    - No detection
    - No background fusion
    - No RealESRGAN
    """

    def __init__(self, device="cuda", fidelity_weight=0.7):
        self.device = torch.device(device)
        self.w = fidelity_weight

        # ------------------ build CodeFormer network ------------------
        self.net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)

        # ------------------ load pretrained weights ------------------
        ckpt_path = load_file_from_url(
            url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            model_dir=f"{CODEFORMER_ROOT}/weights/CodeFormer",
            progress=True,
        )

        checkpoint = torch.load(ckpt_path, map_location="cpu")["params_ema"]
        self.net.load_state_dict(checkpoint, strict=True)
        self.net.eval()

        print(f"[CodeFormer] Loaded on {self.device}, w={self.w}")

    @torch.no_grad()
    def restore_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Args:
            face_img: np.ndarray (512, 512, 3), RGB, uint8, aligned face
        Returns:
            restored face: np.ndarray (512, 512, 3), RGB, uint8
        """

        assert (
            face_img.shape[0] == 512 and face_img.shape[1] == 512
        ), "CodeFormer expects aligned 512x512 face"

        # --- to tensor ---
        face_t = img2tensor(face_img / 255.0, bgr2rgb=False, float32=True)
        normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        face_t = face_t.unsqueeze(0).to(self.device)

        # --- forward ---
        output = self.net(face_t, w=self.w, adain=True)[0]

        # --- back to image ---
        restored = tensor2img(output, rgb2bgr=False, min_max=(-1, 1))
        return restored.astype("uint8")
