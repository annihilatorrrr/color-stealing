import argparse
import os

from src.chaos_game import build_fractal
from src.io import most_recent_file
from src.render import cmap_dict, render_fractal


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--resolution', '-r', type=int, default=2 ** 10,
        help='Resolution of the created image. The image is square, so provide only one number.')
    parser.add_argument(
        '--batch_size', '-b', type=int, default=128,
        help='Number of points calculated at each iteration. '
        'For fractal flame, should be <= 4 for better visuals.')
    parser.add_argument(
        '--n_iter', '-i', type=int, default=int(5e4),
        help='Number of iterations for the chaos game algorithm')
    parser.add_argument(
        '--n_ignore', '-ni', type=int, default=int(2e2),
        help='Number of initial iterations that will not be rendered.')
    parser.add_argument(
        '--cmap', '-cm', type=str, default='i', choices=list(cmap_dict.keys()),
        help='Matplotlib cmap for plotting/saving the image. Ignored if -f or -c is given.')
    parser.add_argument(
        '--plot', '-p', action='store_true',
        help='If given, the image will be rendered with plt.imshow.')
    parser.add_argument(
        '--save', '-s', nargs='?', const=True,
        help='Usage: If not given, the image will not be saved.'
        'If run as "-s", the image & IFS will be stored with the current timestamp as names.'
        'If run as "-s <path>", the image will be stored as <path>.png and the IFS as <path>.csv.')
    parser.add_argument(
        '--load', '-l', type=str, default='',
        help='Path to stored csv IFS file. Usage: -l <path>.')
    parser.add_argument(
        '--mutate', '-m', type=str, default='',
        help='Path to stored csv IFS file that will be loaded & mutated. Usage: -m <path>.')
    parser.add_argument(
        '--no_tqdm', '-nt', action='store_true',
        help='If given, the tqdm progress bar will not be used.')
    parser.add_argument(
        '--dir', '-d', type=str, default='creations',
        help='Directory where the image & csv file will be stored.')
    parser.add_argument(
        '--color_steal', '-c', type=str, default='',
        help='Path to an image that will be used for color stealing. Usage: -c <path>.')
    parser.add_argument(
        '--flame', '-f', action='store_true',
        help='If given, the Fractal Flame Algorithm will be used.')
    parser.add_argument(
        '--gamma', '-g', type=float, default=20.,
        help='Gamma parameter for the fractal frame algorithm.')
    parser.add_argument(
        '--sup', '-su', type=int, default=3,
        help='Supersampling ratio for the fractal flame algorithm.')

    args = parser.parse_args()
    args.cmap = cmap_dict[args.cmap]

    assert args.save or args.plot, 'Provide either -s or -p to run.'
    assert not (args.color_steal and args.flame), \
        'Cannot use both color stealing and fractal flame. Pick one.'

    if args.color_steal:
        assert os.path.isfile(args.color_steal), \
            f'"{args.color_steal}" is not a valid image.'

    if args.load == 'r':
        args.load = most_recent_file(args)
    if args.mutate == 'r':
        args.mutate = most_recent_file(args)

    assert args.gamma > 0, 'Gamma must be positive.'
    assert args.sup >= 1, 'Sup must be at least one.'

    return args


def main():
    args = parse_args()
    canvas, ifs = build_fractal(args)
    render_fractal(canvas=canvas, ifs=ifs, args=args)


if __name__ == '__main__':
    main()
