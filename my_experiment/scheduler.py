def transformer_lr(step, d_model, warmup, factor=1.0):
    if step == 0:
        step = 1
    return factor * (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)
