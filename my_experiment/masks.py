import torch

from .constants import PAD


def subsequent_mask(size, device):
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def make_src_mask(src):
    return (src != PAD).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt):
    pad_mask = (tgt != PAD).unsqueeze(1).unsqueeze(2)
    seq_mask = subsequent_mask(tgt.size(1), tgt.device)
    return pad_mask & seq_mask
