import torch.nn as nn

from .masks import make_src_mask, make_tgt_mask
from .positional_encoding import TokenEmbedding, build_positional_encoding
from .layers import Encoder, Decoder


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        dropout,
        pe_type,
        residual_type,
        max_len,
    ):
        super().__init__()

        self.src_embedding = TokenEmbedding(vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(vocab_size, d_model)

        self.src_pe = build_positional_encoding(pe_type, d_model, dropout, max_len)
        self.tgt_pe = build_positional_encoding(pe_type, d_model, dropout, max_len)

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout, residual_type)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout, residual_type)
        self.generator = nn.Linear(d_model, vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        x = self.src_pe(self.src_embedding(src))
        return self.encoder(x, src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.tgt_pe(self.tgt_embedding(tgt))
        return self.decoder(x, memory, src_mask, tgt_mask)

    def forward(self, src, tgt_in):
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_in)

        memory = self.encode(src, src_mask)
        out = self.decode(tgt_in, memory, src_mask, tgt_mask)

        return self.generator(out)
