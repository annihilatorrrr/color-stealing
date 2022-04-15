import numpy as np
import tqdm

from src.ifs import build_ifs, sample_ifs
from src.io import read_image


def loop_wrapper(x, args):
    return x if args.no_tqdm else tqdm.tqdm(x)


def find_min_max(point_list):
    min_, max_ = point_list[0][0]

    for batch in point_list:
        min_ = np.minimum(min_, batch.min(axis=0))
        max_ = np.maximum(max_, batch.max(axis=0))
    min_, max_ = min_[None, ...], max_[None, ...]

    return min_, max_


def binary_fractal(args):
    w, b, p = build_ifs(args)
    fractal_set = []
    batch = np.random.uniform(size=[args.batch_size, 2]).astype(np.float32)

    for step in loop_wrapper(range(args.n_iter), args=args):
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

    min_, max_ = find_min_max(fractal_set)

    for batch in fractal_set:
        batch = (batch - min_) / (max_ - min_)
        batch = np.clip(
            (batch * args.resolution).astype(int), a_min=0, a_max=args.resolution - 1)

        canvas[batch[:, 0], batch[:, 1]] = True

    canvas = np.rot90(canvas)

    return canvas, (w, b, p)


def color_stealing(args):
    image = read_image(args.steal)
    w_draw, b_draw, p_draw = build_ifs(args)
    w_color, b_color, _ = sample_ifs(w_draw.shape[0])
    buffer_draw, buffer_color = [], []

    batch_draw = np.random.uniform(
        size=[args.batch_size, 2]).astype(np.float32)
    batch_color = np.random.uniform(
        size=[args.batch_size, 2]).astype(np.float32)

    for step in loop_wrapper(range(args.n_iter), args=args):
        chosen_idxs = np.random.choice(
            w_draw.shape[0], size=args.batch_size, p=p_draw)
        next_batch_draw, next_batch_color = [], []

        for ifs_idx in range(w_draw.shape[0]):
            next_batch_draw.append(
                np.matmul(batch_draw[chosen_idxs == ifs_idx], w_draw[ifs_idx]) + b_draw[ifs_idx])
            next_batch_color.append(
                np.matmul(batch_color[chosen_idxs == ifs_idx], w_color[ifs_idx]) + b_color[ifs_idx])

        batch_draw = np.concatenate(next_batch_draw, axis=0)
        batch_color = np.concatenate(next_batch_color, axis=0)

        if step > args.n_ignore:
            buffer_draw.append(batch_draw)
            buffer_color.append(batch_color)

    canvas_shape = [args.resolution, args.resolution, image.shape[2]]
    canvas = np.full(shape=canvas_shape, fill_value=0, dtype=image.dtype)

    min_draw, max_draw = find_min_max(buffer_draw)
    min_color, max_color = find_min_max(buffer_color)
    image_res = np.array(image.shape[:2])

    for batch_draw, batch_color in zip(buffer_draw, buffer_color):
        batch_draw = (batch_draw - min_draw) / (max_draw - min_draw)
        batch_draw = np.clip(
            (batch_draw * args.resolution).astype(int), a_min=0, a_max=args.resolution - 1)

        batch_color = (batch_color - min_color) / (max_color - min_color)
        batch_color = np.clip(
            (batch_color * image_res).astype(int), a_min=0, a_max=image_res - 1)

        canvas[
            batch_draw[:, 0], batch_draw[:, 1]] = image[batch_color[:, 0], batch_color[:, 1]]

    canvas = np.rot90(canvas)

    return canvas, (w_draw, b_draw, p_draw)


def flame_fractal(args):
    pass


def build_fractal(args):
    assert not (args.steal and args.flame)

    if args.steal:
        fractal = color_stealing(args)
    elif args.flame:
        fractal = flame_fractal(args)
    else:
        fractal = binary_fractal(args)

    return fractal
