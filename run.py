import argparse
import os

from src import hardcoded
from src.chaos_game import build_fractal
from src.io import most_recent_file
from src.render import cmap_dict, render_fractal


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--fixed', '-fi', type=str, default='')
    parser.add_argument(
        '--resolution', '-r', type=int, default=2 ** 10)
    parser.add_argument(
        '--batch_size', '-b', type=int, default=2 ** 8)
    parser.add_argument(
        '--n_iter', '-i', type=int, default=int(1e5))
    parser.add_argument(
        '--n_ignore', '-ni', type=int, default=int(2e2))
    parser.add_argument(
        '--cmap', '-cm', type=str, default='i', choices=list(cmap_dict.keys()))
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
    parser.add_argument(
        '--color_steal', '-c', type=str, default='')
    parser.add_argument(
        '--flame', '-f', action='store_true')
    parser.add_argument(
        '--gamma', '-g', type=float, default=2.)
    parser.add_argument(
        '--sup', '-su', type=int, default=3)

    args = parser.parse_args()
    args.cmap = cmap_dict[args.cmap]

    assert args.save or args.plot
    assert not (args.color_steal and args.flame)

    if args.color_steal:
        assert os.path.isfile(args.color_steal)

    if args.fixed:
        assert args.fixed in hardcoded.w
    if args.load == 'r':
        args.load = most_recent_file(args)
    if args.mutate == 'r':
        args.mutate = most_recent_file(args)

    assert args.gamma > 0 and args.sup >= 1

    return args


def main():
    args = parse_args()
    canvas, ifs = build_fractal(args)
    render_fractal(canvas=canvas, ifs=ifs, args=args)


if __name__ == '__main__':
    main()
