import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, residual_type):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.residual_type = residual_type

    def forward(self, x, sublayer):
        if self.residual_type == "standard":
            return self.norm(x + self.dropout(sublayer(x)))

        if self.residual_type == "no_residual":
            return self.norm(self.dropout(sublayer(x)))

        if self.residual_type == "no_layernorm":
            return x + self.dropout(sublayer(x))

        if self.residual_type == "no_residual_no_layernorm":
            return self.dropout(sublayer(x))

        if self.residual_type == "pre_norm":
            return x + self.dropout(sublayer(self.norm(x)))

        raise ValueError(f"Unknown residual_type: {self.residual_type}")


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, residual_type):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout, residual_type)
        self.sublayer2 = SublayerConnection(d_model, dropout, residual_type)

    def forward(self, x, src_mask):
        x = self.sublayer1(x, lambda z: self.self_attn(z, z, z, src_mask))
        x = self.sublayer2(x, self.ffn)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, residual_type):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.src_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.sublayer1 = SublayerConnection(d_model, dropout, residual_type)
        self.sublayer2 = SublayerConnection(d_model, dropout, residual_type)
        self.sublayer3 = SublayerConnection(d_model, dropout, residual_type)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer1(x, lambda z: self.self_attn(z, z, z, tgt_mask))
        x = self.sublayer2(x, lambda z: self.src_attn(z, memory, memory, src_mask))
        x = self.sublayer3(x, self.ffn)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout, residual_type):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout, residual_type)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout, residual_type):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout, residual_type)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x
