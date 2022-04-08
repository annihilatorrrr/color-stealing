import argparse

from src import hardcoded
from src.build import build_fractal
from src.io import most_recent_file
from src.render import cmap_dict, render_fractal


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--fixed', '-f', type=str, default='')
    parser.add_argument(
        '--resolution', '-r', type=int, default=2 ** 10)
    parser.add_argument(
        '--batch_size', '-b', type=int, default=2 ** 8)
    parser.add_argument(
        '--n_iter', '-i', type=int, default=int(1e5))
    parser.add_argument(
        '--n_ignore', '-ni', type=int, default=int(2e2))
    parser.add_argument(
        '--cmap', '-c', type=str, default='i',
        choices=list(cmap_dict.keys()))
    parser.add_argument(
        '--plot', '-p', action='store_true')
    parser.add_argument(
        '--save', '-s', action='store_true')
    parser.add_argument(
        '--load', '-l', type=str, default='')
    parser.add_argument(
        '--mutate', '-m', type=str, default='')
    parser.add_argument(
        '--no_tqdm', '-nt', action='store_true')
    parser.add_argument(
        '--dir', '-d', type=str, default='creations')

    args = parser.parse_args()

    assert args.save or args.plot

    if args.plot or args.save:
        assert args.cmap in cmap_dict
        args.cmap = cmap_dict[args.cmap]
    if args.fixed:
        assert args.fixed in hardcoded.w
    if args.load == 'r':
        args.load = most_recent_file(args)
    if args.mutate == 'r':
        args.mutate = most_recent_file(args)

    return args


def main():
    args = parse_args()
    canvas, ifs = build_fractal(args)
    render_fractal(canvas=canvas, ifs=ifs, args=args)


if __name__ == '__main__':
    main()
