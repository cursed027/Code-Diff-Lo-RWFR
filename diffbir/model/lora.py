# diffbir/model/lora.py
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)

        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale
