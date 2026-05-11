import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

from .constants import PAD


def plot_loss_curves(results_dir):
    files = glob.glob(os.path.join(results_dir, "*_history.csv"))
    if not files:
        return

    plt.figure(figsize=(9, 5))
    for file in files:
        df = pd.read_csv(file)
        if "valid_loss" in df.columns:
            plt.plot(df["epoch"], df["valid_loss"], label=df["experiment"].iloc[0])

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "valid_loss_curves.png"), dpi=200)
    plt.close()


def plot_attention_heatmap(model, loader, device, save_path):
    model.eval()
    src, tgt = next(iter(loader))
    src = src[:1].to(device)
    tgt = tgt[:1].to(device)

    tgt_in = tgt[:, :-1]
    _ = model(src, tgt_in)

    attn = model.decoder.layers[-1].src_attn.last_attn
    if attn is None:
        return

    src_len = int((src[0] != PAD).sum().item())
    tgt_len = int((tgt_in[0] != PAD).sum().item())

    mat = attn[0, 0, :tgt_len, :src_len].cpu().numpy()

    x_labels = [str(int(x)) for x in src[0, :src_len].cpu()]
    y_labels = [str(int(x)) for x in tgt_in[0, :tgt_len].cpu()]

    plt.figure(figsize=(7, 5))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.xticks(range(src_len), x_labels)
    plt.yticks(range(tgt_len), y_labels)
    plt.xlabel("Source tokens")
    plt.ylabel("Decoder input tokens")
    plt.title("Decoder-Source Attention Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
