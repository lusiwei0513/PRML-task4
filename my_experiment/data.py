import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .constants import PAD, BOS, EOS


class ReverseDataset(Dataset):
    def __init__(self, n_samples, min_len, max_len, vocab_size, seed):
        self.samples = []
        rng = np.random.default_rng(seed)

        for _ in range(n_samples):
            length = int(rng.integers(min_len, max_len + 1))
            seq = rng.integers(3, vocab_size, size=length).tolist()

            src = seq + [PAD] * (max_len - length)
            rev = list(reversed(seq))
            tgt = [BOS] + rev + [EOS] + [PAD] * (max_len - length)

            self.samples.append(
                (
                    torch.tensor(src, dtype=torch.long),
                    torch.tensor(tgt, dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_loaders(args):
    train_set = ReverseDataset(
        args.train_size,
        args.train_min_len,
        args.train_max_len,
        args.vocab_size,
        args.seed,
    )
    valid_set = ReverseDataset(
        args.valid_size,
        args.train_min_len,
        args.train_max_len,
        args.vocab_size,
        args.seed + 1,
    )
    test_id_set = ReverseDataset(
        args.test_size,
        args.train_min_len,
        args.train_max_len,
        args.vocab_size,
        args.seed + 2,
    )
    test_ood_set = ReverseDataset(
        args.test_size,
        args.ood_min_len,
        args.ood_max_len,
        args.vocab_size,
        args.seed + 3,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_id_loader = DataLoader(test_id_set, batch_size=args.batch_size, shuffle=False)
    test_ood_loader = DataLoader(test_ood_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, valid_loader, test_id_loader, test_ood_loader
