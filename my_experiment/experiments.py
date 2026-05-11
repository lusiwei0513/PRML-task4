import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .constants import PAD
from .utils import set_seed
from .model import Seq2SeqTransformer
from .engine import train_one_epoch, evaluate
from .scheduler import transformer_lr


def build_model(args, pe_type, residual_type, device):
    max_len = args.ood_max_len + 2

    model = Seq2SeqTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pe_type=pe_type,
        residual_type=residual_type,
        max_len=max_len,
    )

    return model.to(device)


def run_experiment(exp_name, pe_type, residual_type, args, loaders, device):
    print(f"\n===== {exp_name} | PE={pe_type} | residual={residual_type} =====")

    train_loader, valid_loader, test_id_loader, test_ood_loader = loaders

    set_seed(args.seed)
    model = build_model(args, pe_type, residual_type, device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: transformer_lr(step, args.d_model, args.warmup, args.lr_factor),
    )

    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, grad_norm = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, args.clip_grad
        )

        valid_stats = evaluate(model, valid_loader, criterion, device)

        row = {
            "experiment": exp_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_stats["loss"],
            "valid_teacher_token_acc": valid_stats["teacher_token_acc"],
            "valid_greedy_token_acc": valid_stats["greedy_token_acc"],
            "valid_sequence_acc": valid_stats["sequence_acc"],
            "grad_norm": grad_norm,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={valid_stats['loss']:.4f} | "
            f"valid_seq_acc={valid_stats['sequence_acc']:.4f} | "
            f"grad_norm={grad_norm:.3f}"
        )

        if not np.isfinite(train_loss):
            break

    test_id = evaluate(model, test_id_loader, criterion, device)
    test_ood = evaluate(model, test_ood_loader, criterion, device)

    elapsed = time.time() - start

    summary = {
        "experiment": exp_name,
        "pe_type": pe_type,
        "residual_type": residual_type,
        "epochs": len(history),
        "time_sec": elapsed,
        "test_id_loss": test_id["loss"],
        "test_id_teacher_token_acc": test_id["teacher_token_acc"],
        "test_id_greedy_token_acc": test_id["greedy_token_acc"],
        "test_id_sequence_acc": test_id["sequence_acc"],
        "test_ood_loss": test_ood["loss"],
        "test_ood_teacher_token_acc": test_ood["teacher_token_acc"],
        "test_ood_greedy_token_acc": test_ood["greedy_token_acc"],
        "test_ood_sequence_acc": test_ood["sequence_acc"],
    }

    history_df = pd.DataFrame(history)
    history_path = os.path.join(args.results_dir, f"{exp_name}_history.csv")
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    return model, summary


def get_experiments(mode):
    pe_exps = [
        ("pe_none", "none", "standard"),
        ("pe_simple", "simple", "standard"),
        ("pe_learned", "learned", "standard"),
        ("pe_sinusoidal", "sinusoidal", "standard"),
    ]

    residual_exps = [
        ("res_standard", "sinusoidal", "standard"),
        ("res_no_residual", "sinusoidal", "no_residual"),
        ("res_no_layernorm", "sinusoidal", "no_layernorm"),
        ("res_no_residual_no_layernorm", "sinusoidal", "no_residual_no_layernorm"),
    ]

    if mode == "baseline":
        return [("baseline", "sinusoidal", "standard")]
    if mode == "pe":
        return pe_exps
    if mode == "residual":
        return residual_exps
    if mode == "all":
        return pe_exps + residual_exps[1:]

    raise ValueError(f"Unknown mode: {mode}")
