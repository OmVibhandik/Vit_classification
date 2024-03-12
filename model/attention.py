import torch.nn as nn
import torchvision
import torch


class Attention(nn.Module):
    def __init__(self, embed_dim, heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.dropout = nn.Dropout(dropout)
        assert (
            self.head_dim * heads == embed_dim
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.keys = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.queries = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(self, query, keys, values, attn_mask=None, key_padding_mask=None):
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if attn_mask is None:
            energy = energy / (self.embed_dim ** (1 / 2))
        else:
            energy = (energy + attn_mask) / (self.embed_dim ** (1 / 2))
        if key_padding_mask is None:
            attention = torch.softmax(energy, dim=3)
        else:
            attention = torch.softmax(energy + key_padding_mask, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)

        avg_attn = attention.sum(dim=1)
        avg_attn /= self.heads

        return self.dropout(out), attention
