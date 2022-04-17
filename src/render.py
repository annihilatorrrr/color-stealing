import datetime
import os
import matplotlib.pyplot as plt

from src.io import img_subdir, ifs_subdir, save_ifs


cmap_dict = {
    cmap[0]: cmap for cmap in ['inferno', 'plasma', 'magma', 'cividis', 'viridis']
}


def render_fractal(canvas, ifs, args):
    """Renders or/and saves the constructed image.

    Args:
        canvas (np.array [res, res] or [res, res, 3]): The generated image.
        ifs (tuple of (np.array, np.array, np.array)): w [n, 2, 2], 
            b [n, 1, 2], p [n].
        args (argparse.Namespace): Arguments for the fractal construction. See run.py
            for more details.
    """
    if args.plot:
        plt.figure(figsize=(17, 17))
        plt.imshow(canvas, cmap=args.cmap)
        plt.axis('off')
        plt.show()

    img_dir = os.path.join(args.dir, img_subdir)
    ifs_dir = os.path.join(args.dir, ifs_subdir)

    if args.save:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ifs_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        img_path = os.path.join(img_dir, f'{timestamp}.png')
        ifs_path = os.path.join(ifs_dir, f'{timestamp}.csv')

        plt.imsave(fname=img_path, arr=canvas, cmap=args.cmap)
        save_ifs(ifs=ifs, path=ifs_path)
