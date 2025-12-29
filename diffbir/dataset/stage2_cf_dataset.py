from typing import Tuple
import os
import random
import numpy as np
from PIL import Image
import torch.utils.data as data


class Stage2CFDataset(data.Dataset):
    def __init__(self, hq_dir, lq_dir, cf_dir, out_size=512):
        self.hq_dir = hq_dir
        self.lq_dir = lq_dir
        self.cf_dir = cf_dir
        self.out_size = out_size

        self.files = sorted(os.listdir(hq_dir))
        assert len(self.files) > 0, "Empty dataset"

    def _load_img(self, path, normalize):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.out_size, self.out_size), Image.BICUBIC)
        img = np.array(img).astype(np.float32) / 255.0

        if normalize == "neg1_1":
            img = img * 2.0 - 1.0  # [-1,1]

        return img

    def __getitem__(self, idx) -> Tuple:
        name = self.files[idx]

        gt = self._load_img(os.path.join(self.hq_dir, name), "neg1_1")
        lq = self._load_img(os.path.join(self.lq_dir, name), "0_1")
        cf = self._load_img(os.path.join(self.cf_dir, name), "0_1")

        prompt = ""  # faces → empty prompt is safest

        return gt, lq, cf, prompt

    def __len__(self):
        return len(self.files)
