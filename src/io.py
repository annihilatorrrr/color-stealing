import numpy as np
import os
import pathlib

img_dir = 'images'
ifs_dir = 'ifs'


def save_ifs(ifs, path):
    w, b, p = ifs
    w_b = np.concatenate([w, b], axis=1)

    with open(path, 'w') as file:
        file.truncate()
        file.write(f'{w_b.shape[0]}\n')

        for slice in w_b:
            np.savetxt(file, slice)
            file.write('\n')

        np.savetxt(file, p)


def load_ifs(path):
    assert os.path.isfile(path)

    with open(path, 'r') as file:
        n_ifs = int(file.readline())

        w_b, p = [], []
        for step in range(1, 4 * n_ifs + 1):
            row = file.readline()
            if step % 4:
                w_b.append(row)

        for step in range(n_ifs):
            prob = float(file.readline())
            assert prob > 0
            p.append(prob)

    w_b = np.array([[float(x) for x in row.split()] for row in w_b])
    w_b = w_b.reshape([n_ifs, 3, 2])
    w, b = np.split(w_b, [2], axis=1)

    p = np.array(p)
    p /= p.sum()

    return w, b, p


def most_recent_file():
    all_files = pathlib.Path(ifs_dir).rglob('*.csv')
    most_recent = max(all_files, key=os.path.getmtime)

    return most_recent
