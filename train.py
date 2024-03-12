import torch
import torch.nn as nn
from torch.optim import Adam
from transformer import VIT

model = VIT(3, 768, 16, 1000, 3072, 0.0, 12, 12)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(params=model.parameters(),
                 lr=8e-4,
                 betas=(0.9, 0.999),
                 weight_decay=0.1)
