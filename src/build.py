import numpy as np
import torch
import tqdm

from src.ifs import build_ifs


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_np(args):

    w, b, p = build_ifs(args)
    fractal_set = []
    batch = np.random.uniform(size=[args.batch_size, 2]).astype(np.float32)

    def wrapper(x): return x if args.no_tqdm else tqdm.tqdm(x)

    for step in wrapper(range(args.n_iter)):
        chosen_idxs = np.random.choice(w.shape[0], size=args.batch_size, p=p)
        next_batch = []

        for ifs_idx in range(w.shape[0]):
            next_batch.append(
                np.matmul(batch[chosen_idxs == ifs_idx], w[ifs_idx]) + b[ifs_idx])

        batch = np.concatenate(next_batch, axis=0)

        if step > args.n_ignore:
            fractal_set.append(batch)

    canvas = np.full(
        [args.resolution, args.resolution], fill_value=False, dtype=bool)

    min_, max_ = fractal_set[0][0]
    for batch in fractal_set:
        min_ = np.minimum(min_, batch.min(axis=0))
        max_ = np.maximum(max_, batch.max(axis=0))
    min_, max_ = min_[None, ...], max_[None, ...]

    for batch in fractal_set:
        batch = (batch - min_) / (max_ - min_)
        batch = np.clip(
            (batch * args.resolution).astype(int), a_min=0, a_max=args.resolution - 1)

        canvas[batch[:, 0], batch[:, 1]] = True

    canvas = np.rot90(canvas)

    return canvas, (w, b, p)


def build_tr(args):

    device = get_device()
    w, b, p = [torch.tensor(x).to(device) for x in build_ifs(args)]
    fractal_set = []
    batch = torch.rand(
        size=[args.batch_size, 2], dtype=torch.float32, device=device)

    def wrapper(x): return x if args.no_tqdm else tqdm.tqdm(x)

    for step in wrapper(range(args.n_iter)):
        chosen_idxs = torch.multinomial(
            input=p, num_samples=args.batch_size, replacement=True).to(device)
        next_batch = []
        #print(batch.dtype, w.dtype, b.dtype)

        for ifs_idx in range(w.shape[0]):
            next_batch.append(
                torch.matmul(batch[chosen_idxs == ifs_idx], w[ifs_idx]) + b[ifs_idx])

        batch = torch.cat(next_batch, dim=0)

        if step > args.n_ignore:
            fractal_set.append(batch)

    canvas = torch.full(
        size=[args.resolution, args.resolution], fill_value=False, dtype=bool, device=device)

    min_, max_ = fractal_set[0][0]
    for batch in fractal_set:
        min_ = torch.minimum(min_, batch.amin(dim=0))
        max_ = torch.maximum(max_, batch.amax(dim=0))
    min_, max_ = min_[None, ...], max_[None, ...]

    for batch in fractal_set:
        batch = (batch - min_) / (max_ - min_)
        batch = torch.clip(
            (batch * args.resolution).long(), min=0, max=args.resolution - 1)

        canvas[batch[:, 0], batch[:, 1]] = True

    canvas = np.rot90(canvas.cpu().numpy())
    w, b, p = [x.cpu().numpy() for x in [w, b, p]]

    return canvas, (w, b, p)


def build_fractal(args):
    return build_tr(args) if torch.cuda.is_available() else build_np(args)
