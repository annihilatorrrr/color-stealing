import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

img_subdir = 'images'
ifs_subdir = 'ifs'


def save_ifs(ifs, path):
    """Saves a given IFS into a csv file.

    Args:
        ifs (tuple of (np.array, np.array, np.array)): w [n, 2, 2], 
            b [n, 1, 2], p [n].
        path (str): Where to store the IFS.
    """
    w, b, p = ifs
    w_b = np.concatenate([w, b], axis=1)

    with open(path, 'w') as file:
        file.truncate()
        file.write(f'{w_b.shape[0]}\n')

        for slice in w_b:
            np.savetxt(file, slice)
            file.write('\n')

        np.savetxt(file, p)


def read_ifs(path):
    """Reads an IFS from a given csv file.

    Args:
        path (str): Path to the csv file.

    Returns:
        np.array [n, 2, 2]: Matrix w. 
        np.array [n, 1, 2]: Bias b.
        np.array [n]: Probability distribution p.
    """
    assert os.path.isfile(path), f'"{path}" is not a valid IFS.'

    with open(path, 'r') as file:
        n_ifs = int(file.readline())

        w_b, p = [], []
        for step in range(1, 4 * n_ifs + 1):
            row = file.readline()
            if step % 4:
                w_b.append(row)

        for step in range(n_ifs):
            prob = float(file.readline())
            assert prob > 0, 'Probabilities must be positive numbers.'
            p.append(prob)

    w_b = np.array([[float(x) for x in row.split()] for row in w_b])
    w_b = w_b.reshape([n_ifs, 3, 2])
    w, b = np.split(w_b, [2], axis=1)

    p = np.array(p)
    p /= p.sum()

    return w, b, p


def most_recent_file(args):
    """Finds the most recently modified IFS file.

    Args:
        args (argparse.Namespace): Arguments for the fractal construction. See run.py
            for more details.

    Returns:
        str: Path to the most recently modified IFS file.
    """
    csv_dir = os.path.join(args.dir, ifs_subdir)
    all_files = pathlib.Path(csv_dir).rglob('*.csv')
    assert all_files, f'Directory "{csv_dir}" does not contain any csv files.'
    most_recent = max(all_files, key=os.path.getmtime)

    return most_recent


def read_image(path):
    """Reads a given image.

    Args:
        path (str): Path to the image.

    Returns:
        np.array: Image as NumPy array.
    """
    assert os.path.isfile(path), f'"{path}" is not a valid image.'
    image = plt.imread(path)
    assert len(image.shape) == 3 and image.shape[2] >= 3, \
        'Image must have at least 3 channels.'
    if image.shape[2] > 3:
        image = image[..., :3]

    return image
