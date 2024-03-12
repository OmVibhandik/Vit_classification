import torchvision
from torchvision import transforms
import torch.nn as nn
import torch
import math


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_dim, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embedding_dim,
                              patch_size, patch_size)
        self.flatten = nn.Flatten(2, 3)

    def forward(self, img):
        img = self.conv(img)
        img = self.flatten(img)
        img = img.permute(0, 2, 1)
        return img


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, embedding_dim)

    def forward(self, x):
        return self.class_embed(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, patch_size, embedding_dim):
        super().__init__()
        self.position = nn.Embedding(patch_size**2 + 1, embedding_dim)

    def forward(self, x):
        batch_size, max_len, _ = x.shape
        positions = torch.arange(0, max_len).expand(batch_size, max_len)

        pos = self.position(positions)
        # print(pos.shape)
        x = x + pos
        return x


class Embeddings(nn.Module):
    def __init__(self, in_channels, embedding_dim, patch_size, num_classes):
        super().__init__()

        self.patch_embedder = PatchEmbedding(
            in_channels, embedding_dim, patch_size)
        self.positional_embedder = PositionalEmbedding(
            patch_size, embedding_dim)
        self.class_embedder = ClassEmbedding(num_classes, embedding_dim)

    def forward(self, x, label):
        patch_embed = self.patch_embedder(x)
        print(patch_embed.shape)
        class_embed = self.class_embedder(label)
        patch_embed = torch.cat((class_embed, patch_embed), dim=1)
        total_embed = self.positional_embedder(patch_embed)

        return total_embed


if __name__ == "__main__":
    x = torch.rand(1, 3, 128, 128)
    label = torch.LongTensor([[1]])
    embedder = Embeddings(3, 768, 16, 1000)
    print(embedder(x, label).shape)
