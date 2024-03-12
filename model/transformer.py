import torch
import torch.nn as nn
from embeddings import Embeddings
from encoder import Encoder
from classifier import Classifier


class VIT(nn.Module):
    def __init__(self, in_channels, embedding_dim, patch_size, num_classes, mlp_size, p_dropout, num_heads, num_layers):
        super().__init__()

        self.embedder = Embeddings(
            in_channels, embedding_dim, patch_size, num_classes)
        self.encoder = Encoder(mlp_size, embedding_dim,
                               num_heads, num_layers, p_dropout)
        self.classifer = Classifier(embedding_dim, num_classes)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, label):
        x = self.dropout(self.embedder(x, label))
        x = self.encoder(x)
        x = self.classifer(x[:, 0])
        return x


# img = torch.rand(1, 3, 256, 256)
# label = torch.tensor([[1]])

# vit = VIT(3, 768, 16, 1000, 3072, 0.1, 12, 12)
# out = vit(img, label)
# print(out.shape)
