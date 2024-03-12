import torch.nn as nn
import torchvision
from attention import Attention
from mlp import MLPBlock


class EncoderLayer(nn.Module):
    def __init__(self, mlp_size, embedding_dim, num_heads, p_dropout):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.attention = Attention(embedding_dim, num_heads, p_dropout)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(mlp_size, embedding_dim, p_dropout)

    def forward(self, x):
        x_norm1 = self.layernorm1(x)
        out, _ = self.attention(x_norm1, x_norm1, x_norm1)
        x = out + x
        x_norm2 = self.layernorm2(x)
        x_fin = self.mlp(x_norm2)
        x = x_fin + x
        return x
