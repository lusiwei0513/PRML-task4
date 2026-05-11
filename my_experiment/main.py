import os
import argparse
import pandas as pd
import torch

from .utils import set_seed
from .data import make_loaders
from .experiments import get_experiments, run_experiment
from .visualization import plot_loss_curves, plot_attention_heatmap


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="all", choices=["baseline", "pe", "residual", "all"])
    parser.add_argument("--results_dir", type=str, default="my_experiment/results")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab_size", type=int, default=33)

    parser.add_argument("--train_size", type=int, default=12000)
    parser.add_argument("--valid_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=1000)

    parser.add_argument("--train_min_len", type=int, default=5)
    parser.add_argument("--train_max_len", type=int, default=20)
    parser.add_argument("--ood_min_len", type=int, default=21)
    parser.add_argument("--ood_max_len", type=int, default=40)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--lr_factor", type=float, default=1.0)
    parser.add_argument("--clip_grad", type=float, default=5.0)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("results_dir:", args.results_dir)

    loaders = make_loaders(args)
    experiments = get_experiments(args.mode)

    summaries = []
    baseline_model = None

    for exp_name, pe_type, residual_type in experiments:
        model, summary = run_experiment(
            exp_name,
            pe_type,
            residual_type,
            args,
            loaders,
            device,
        )
        summaries.append(summary)

        if pe_type == "sinusoidal" and residual_type == "standard" and baseline_model is None:
            baseline_model = model

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(args.results_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plot_loss_curves(args.results_dir)

    if baseline_model is not None:
        _, _, test_id_loader, _ = loaders
        plot_attention_heatmap(
            baseline_model,
            test_id_loader,
            device,
            os.path.join(args.results_dir, "baseline_attention_heatmap.png"),
        )

    print("\nSaved summary to:", summary_path)
    print(summary_df)


if __name__ == "__main__":
    main()
