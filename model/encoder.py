import torch
import torch.nn as nn
import torchvision
from encoder_layer import EncoderLayer
from copy import deepcopy


class Encoder(nn.Module):
    def __init__(self, mlp_size, embedding_dim, num_heads, num_layers, p_dropout):
        super().__init__()
        encoder_layer = EncoderLayer(
            mlp_size, embedding_dim, num_heads, p_dropout)
        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
