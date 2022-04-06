import datetime
import os
import matplotlib.pyplot as plt

from src.io import img_dir, ifs_dir, save_ifs


cmap_dict = {
    cmap[0]: cmap for cmap in ['inferno', 'plasma', 'magma', 'cividis', 'viridis']
}


def render_fractal(canvas, ifs, args):
    if args.plot:
        plt.figure(figsize=(17, 17))
        plt.imshow(canvas, cmap=args.cmap)
        plt.axis('off')
        plt.show()

    if args.save:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ifs_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        img_path = os.path.join(img_dir, f'{timestamp}.png')
        ifs_path = os.path.join(ifs_dir, f'{timestamp}.csv')

        plt.imsave(fname=img_path, arr=canvas, cmap=args.cmap)
        save_ifs(ifs=ifs, path=ifs_path)
