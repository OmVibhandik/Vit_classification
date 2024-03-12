import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()

        self.classifer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        return self.classifer(x)
