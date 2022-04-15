import numpy as np
import skimage.measure
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
    point_buffer = []
    batch = np.random.uniform(size=[args.batch_size, 2]).astype(np.float32)

    for step in loop_wrapper(range(args.n_iter), args=args):
        chosen_idxs = np.random.choice(w.shape[0], size=args.batch_size, p=p)
        next_batch = []

        for ifs_idx in range(w.shape[0]):
            next_batch.append(
                np.matmul(batch[chosen_idxs == ifs_idx], w[ifs_idx]) + b[ifs_idx])

        batch = np.concatenate(next_batch, axis=0)

        if step > args.n_ignore:
            point_buffer.append(batch)

    canvas = np.full(
        [args.resolution, args.resolution], fill_value=False, dtype=bool)

    min_, max_ = find_min_max(point_buffer)

    for batch in point_buffer:
        batch = (batch - min_) / (max_ - min_)
        batch = np.clip(
            (batch * args.resolution).astype(int), a_min=0, a_max=args.resolution - 1)

        canvas[batch[:, 0], batch[:, 1]] = True

    canvas = np.rot90(canvas)

    return canvas, (w, b, p)


def color_steal(args):
    """Color Stealing Algorithm:
    https://maths-people.anu.edu.au/~barnsley/pdfs/fractal_tops.pdf
    """
    image = read_image(args.color_steal)
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
    canvas = np.zeros(shape=canvas_shape, dtype=image.dtype)

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
    w, b, p = build_ifs(args)
    colors = np.random.uniform(size=[w.shape[0], 3])
    point_buffer, idx_buffer = [], []
    point_batch = np.random.uniform(
        size=[args.batch_size, 2]).astype(np.float32)

    for step in loop_wrapper(range(args.n_iter), args=args):
        idx_batch = np.random.choice(w.shape[0], size=args.batch_size, p=p)
        next_point_batch = []

        for ifs_idx in range(w.shape[0]):
            next_point_batch.append(
                np.matmul(point_batch[idx_batch == ifs_idx], w[ifs_idx]) + b[ifs_idx])

        point_batch = np.concatenate(next_point_batch, axis=0)

        if step > args.n_ignore:
            point_buffer.append(point_batch)
            idx_buffer.append(idx_batch)

    sup_res = args.sup * args.resolution
    canvas_shape = [sup_res, sup_res, 3]
    canvas = np.zeros(shape=canvas_shape, dtype=np.float32)
    freq = np.ones(shape=canvas_shape[:-1], dtype=np.float32)

    min_, max_ = find_min_max(point_buffer)

    for point_batch, idx_batch in zip(point_buffer, idx_buffer):
        point_batch = (point_batch - min_) / (max_ - min_)
        point_batch = np.clip(
            (point_batch * sup_res).astype(int), a_min=0, a_max=sup_res - 1)

        freq[point_batch[:, 0], point_batch[:, 1]] += 1
        canvas[point_batch[:, 0], point_batch[:, 1]] = (
            canvas[point_batch[:, 0], point_batch[:, 1]] + colors[idx_batch]) / 2.

    block_size = (args.sup, args.sup, 1)
    freq = skimage.measure.block_reduce(
        freq, block_size=block_size[:2], func=np.mean)
    canvas = skimage.measure.block_reduce(
        canvas, block_size=block_size, func=np.mean)

    eps = 1e-6
    alpha = np.log(freq[..., None]) / np.log(freq.max())

    canvas = canvas * alpha ** (1/args.gamma)

    canvas = np.rot90(canvas)

    return canvas, (w, b, p)


def build_fractal(args):
    assert not (args.color_steal and args.flame)

    if args.color_steal:
        fractal = color_steal(args)
    elif args.flame:
        fractal = flame_fractal(args)
    else:
        fractal = binary_fractal(args)

    return fractal
