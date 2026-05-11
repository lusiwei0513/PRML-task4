import numpy as np
import torch
from tqdm import tqdm

from .constants import PAD, BOS
from .masks import make_src_mask, make_tgt_mask


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, clip_grad):
    model.train()

    total_loss = 0.0
    total_tokens = 0
    grad_norms = []

    for src, tgt in tqdm(loader, leave=False):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        logits = model(src, tgt_in)
        ntokens = (tgt_y != PAD).sum()

        loss_sum = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))
        loss = loss_sum / ntokens

        if not torch.isfinite(loss):
            return float("nan"), float("nan")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()

        total_loss += loss_sum.item()
        total_tokens += ntokens.item()
        grad_norms.append(float(grad_norm))

    return total_loss / total_tokens, float(np.mean(grad_norms))


@torch.no_grad()
def greedy_decode(model, src, max_len, device):
    model.eval()

    src_mask = make_src_mask(src)
    memory = model.encode(src, src_mask)

    ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys)
        out = model.decode(ys, memory, src_mask, tgt_mask)
        logits = model.generator(out[:, -1])
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

    return ys


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    teacher_correct = 0
    teacher_tokens = 0

    greedy_correct = 0
    greedy_tokens = 0
    seq_correct = 0
    seq_total = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        logits = model(src, tgt_in)
        ntokens = (tgt_y != PAD).sum()

        loss_sum = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))
        total_loss += loss_sum.item()
        total_tokens += ntokens.item()

        pred_teacher = torch.argmax(logits, dim=-1)
        mask = tgt_y != PAD

        teacher_correct += ((pred_teacher == tgt_y) & mask).sum().item()
        teacher_tokens += mask.sum().item()

        pred_greedy = greedy_decode(model, src, tgt.size(1), device)
        pred_y = pred_greedy[:, 1:]
        tgt_y = tgt[:, 1:]

        mask = tgt_y != PAD
        greedy_correct += ((pred_y == tgt_y) & mask).sum().item()
        greedy_tokens += mask.sum().item()

        seq_match = ((pred_y == tgt_y) | ~mask).all(dim=1)
        seq_correct += seq_match.sum().item()
        seq_total += src.size(0)

    return {
        "loss": total_loss / total_tokens,
        "teacher_token_acc": teacher_correct / max(1, teacher_tokens),
        "greedy_token_acc": greedy_correct / max(1, greedy_tokens),
        "sequence_acc": seq_correct / max(1, seq_total),
    }
