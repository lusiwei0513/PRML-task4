import math
import torch
import torch.nn as nn

from .constants import PAD


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class NoPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x)


class SimpleAbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.linear = nn.Linear(1, d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).float()
        pos = (pos / max(1, self.max_len)).view(1, -1, 1)
        return self.dropout(x + self.linear(pos))


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(pos))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


def build_positional_encoding(pe_type, d_model, dropout, max_len):
    if pe_type == "none":
        return NoPositionalEncoding(d_model, dropout, max_len)
    if pe_type == "simple":
        return SimpleAbsolutePositionalEncoding(d_model, dropout, max_len)
    if pe_type == "learned":
        return LearnedPositionalEncoding(d_model, dropout, max_len)
    if pe_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, dropout, max_len)
    raise ValueError(f"Unknown pe_type: {pe_type}")
