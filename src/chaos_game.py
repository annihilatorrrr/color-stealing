import numpy as np
import skimage.measure
import tqdm

from src.ifs import build_ifs, sample_ifs
from src.io import read_image


def loop_wrapper(x, args):
    """Wraps an iterator inside tqdm if args.no_tqdm == True.
    """
    return x if args.no_tqdm else tqdm.tqdm(x)


def find_min_max(point_list):
    """Finds min & max inside a list of arrays.

    Args:
        point_list (list of np.array of float): Each array has dimensions
        [n, d] where n is variable and d is shared by all arrays.

    Returns:
        np.array of float [d]: Min vector.
        np.array of float [d]: Max vector.
    """
    min_, max_ = point_list[0][0]

    for batch in point_list:
        min_ = np.minimum(min_, batch.min(axis=0))
        max_ = np.maximum(max_, batch.max(axis=0))
    min_, max_ = min_[None, ...], max_[None, ...]

    return min_, max_


def binary_fractal(args):
    """Constructs a binary fractal using chaos game (ie random iterations).

    Args:
        args (argparse.Namespace): Arguments for the fractal construction. See run.py
            for more details.

    Returns:
        np.array of bool [args.resolution, args.resolution]: Binary array where
            points that belong to the fractal are True.
        tuple (np.array, np.array, np.array): Parameters of the IFS that constructed
            the fractal: w [n, 2, 2], b [n, 1, 2], p [n].
    """
    w, b, p = build_ifs(args)
    coord_buffer = []
    batch = np.random.uniform(size=[args.batch_size, 2]).astype(np.float32)

    # To speed things up, at each iteration, a batch of coordinates (instead of a single
    # point) is calculated. The batches are stored inside a buffer and after the final
    # iteration, the coordinates are normalized to [0, 1]x[0, 1] and plotted on a grid.

    for step in loop_wrapper(range(args.n_iter), args=args):
        ifs_idxs = np.random.choice(w.shape[0], size=args.batch_size, p=p)
        next_batch = [
            np.matmul(batch[ifs_idxs == ifs_idx], w[ifs_idx]) + b[ifs_idx]
            for ifs_idx in range(w.shape[0])
        ]
        batch = np.concatenate(next_batch, axis=0)

        # The initial iterations are ignored as the coordinates have not converged yet.
        if step > args.n_ignore:
            coord_buffer.append(batch)

    canvas = np.full(
        [args.resolution, args.resolution], fill_value=False, dtype=bool)

    min_, max_ = find_min_max(coord_buffer)

    for batch in coord_buffer:
        batch = (batch - min_) / (max_ - min_)
        batch = np.clip(
            (batch * args.resolution).astype(int), a_min=0, a_max=args.resolution - 1)

        canvas[batch[:, 0], batch[:, 1]] = True

    canvas = np.rot90(canvas)

    return canvas, (w, b, p)


def color_steal(args):
    """Constructs a colorful fractal using the Color Stealing Algorithm:

    https://maths-people.anu.edu.au/~barnsley/pdfs/fractal_tops.pdf

    Args:
        args (argparse.Namespace): Arguments for the fractal construction. See run.py
            for more details.

    Returns:
        np.array of (float, int) [args.resolution, args.resolution, image.n_channels]: 
            Array representing the generated fractal image.
        tuple (np.array, np.array, np.array): Parameters of the IFS that constructed
            the fractal: w [n, 2, 2], b [n, 1, 2], p [n].
    """
    image = read_image(args.color_steal)
    w_fractal, b_fractal, p_fractal = build_ifs(args)
    w_color, b_color, _ = sample_ifs(w_fractal.shape[0])
    buffer_fractal, buffer_color = [], []

    batch_fractal = np.random.uniform(
        size=[args.batch_size, 2]).astype(np.float32)
    batch_color = np.random.uniform(
        size=[args.batch_size, 2]).astype(np.float32)

    # 2 IFSs are necessary: one for the constructed fractal and one for the given image
    # whose color patterns are being copied.

    for step in loop_wrapper(range(args.n_iter), args=args):
        ifs_idxs = np.random.choice(
            w_fractal.shape[0], size=args.batch_size, p=p_fractal)
        next_batch_fractal, next_batch_color = [], []

        for ifs_idx in range(w_fractal.shape[0]):
            next_batch_fractal.append(
                np.matmul(batch_fractal[ifs_idxs == ifs_idx], w_fractal[ifs_idx]) + b_fractal[ifs_idx])
            next_batch_color.append(
                np.matmul(batch_color[ifs_idxs == ifs_idx], w_color[ifs_idx]) + b_color[ifs_idx])

        batch_fractal = np.concatenate(next_batch_fractal, axis=0)
        batch_color = np.concatenate(next_batch_color, axis=0)

        # The initial iterations are ignored as the coordinates have not converged yet.
        if step > args.n_ignore:
            buffer_fractal.append(batch_fractal)
            buffer_color.append(batch_color)

    canvas_shape = [args.resolution, args.resolution, image.shape[2]]
    canvas = np.zeros(shape=canvas_shape, dtype=image.dtype)

    min_fractal, max_fractal = find_min_max(buffer_fractal)
    min_color, max_color = find_min_max(buffer_color)
    image_res = np.array(image.shape[:2])

    for batch_fractal, batch_color in zip(buffer_fractal, buffer_color):
        batch_fractal = (batch_fractal - min_fractal) / \
            (max_fractal - min_fractal)
        batch_fractal = np.clip(
            (batch_fractal * args.resolution).astype(int), a_min=0, a_max=args.resolution - 1)

        batch_color = (batch_color - min_color) / (max_color - min_color)
        batch_color = np.clip(
            (batch_color * image_res).astype(int), a_min=0, a_max=image_res - 1)

        canvas[
            batch_fractal[:, 0], batch_fractal[:, 1]] = image[batch_color[:, 0], batch_color[:, 1]]

    canvas = np.rot90(canvas)

    return canvas, (w_fractal, b_fractal, p_fractal)


def fractal_flame(args):
    """Constructs a colorful fractal using a smplified Fractal Flame Algorithm:

    https://flam3.com/flame_draves.pdf

    Args:
        args (argparse.Namespace): Arguments for the fractal construction. See run.py
            for more details.

    Returns:
        np.array of (float, int) [args.resolution, args.resolution, 3]: Array representing 
            the generated fractal image.
        tuple (np.array, np.array, np.array): Parameters of the IFS that constructed
            the fractal: w [n, 2, 2], b [n, 1, 2], p [n].
    """
    w, b, p = build_ifs(args)
    colors = np.random.uniform(size=[w.shape[0], 3])
    coord_buffer, idx_buffer = [], []
    coord_batch = np.random.uniform(
        size=[args.batch_size, 2]).astype(np.float32)

    for step in loop_wrapper(range(args.n_iter), args=args):
        idx_batch = np.random.choice(w.shape[0], size=args.batch_size, p=p)
        next_coord_batch = [
            np.matmul(coord_batch[idx_batch == ifs_idx], w[ifs_idx])
            + b[ifs_idx]
            for ifs_idx in range(w.shape[0])
        ]
        coord_batch = np.concatenate(next_coord_batch, axis=0)

        # The initial iterations are ignored as the coordinates have not converged yet.
        if step > args.n_ignore:
            coord_buffer.append(coord_batch)
            idx_buffer.append(idx_batch)

    # The paper claims that drawing inside a grid with larger resolution followed by
    # downsampling leads to better results.
    sup_res = args.sup * args.resolution
    canvas_shape = [sup_res, sup_res, 3]
    canvas = np.zeros(shape=canvas_shape, dtype=np.float32)
    freq = np.ones(shape=canvas_shape[:-1], dtype=np.float32)

    min_, max_ = find_min_max(coord_buffer)

    for coord_batch, idx_batch in zip(coord_buffer, idx_buffer):
        coord_batch = (coord_batch - min_) / (max_ - min_)
        coord_batch = np.clip(
            (coord_batch * sup_res).astype(int), a_min=0, a_max=sup_res - 1)

        freq[coord_batch[:, 0], coord_batch[:, 1]] += 1
        canvas[coord_batch[:, 0], coord_batch[:, 1]] = (
            canvas[coord_batch[:, 0], coord_batch[:, 1]] + colors[idx_batch]) / 2.

    block_size = (args.sup, args.sup, 1)
    freq = skimage.measure.block_reduce(
        freq, block_size=block_size[:2], func=np.mean)
    canvas = skimage.measure.block_reduce(
        canvas, block_size=block_size, func=np.mean)

    alpha = np.log(freq[..., None]) / np.log(freq.max())

    canvas = canvas * alpha ** (1/args.gamma)

    canvas = np.rot90(canvas)

    return canvas, (w, b, p)


def build_fractal(args):
    assert not (args.color_steal and args.flame), \
        'Cannot use both color stealing and fractal flame. Pick one.'

    if args.color_steal:
        return color_steal(args)
    elif args.flame:
        return fractal_flame(args)
    else:
        return binary_fractal(args)
