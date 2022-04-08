import argparse
import cv2
import functools
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import tqdm

import timm
import torch.nn.functional as F
import torch.nn as nn
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--src', '-s', type=str, required=True)
    parser.add_argument(
        '--tgt', '-t', type=str, default='creations/color')
    parser.add_argument(
        '--n_iter', '-i', type=int, default=int(1e3))
    parser.add_argument(
        '--rep', '-re', type=int, default=1)
    parser.add_argument(
        '--batch_size', '-b', type=int, default=128)
    parser.add_argument(
        '--res', '-r', type=int, default=400)
    parser.add_argument(
        '--optim', '-o', type=str, default='sgd', choices=['sgd', 'adam', 'adamw', 'rmsprop'])
    parser.add_argument(
        '--loss', '-l', type=str, default='logit', choices=['logit', 'ce'])
    parser.add_argument(
        '--lr', '-lr', type=float, default=4e-4)
    parser.add_argument(
        '--weight_decay', '-wd', type=float, default=0.05)
    parser.add_argument(
        '--aux', '-a', type=float, default=0.01)
    parser.add_argument(
        '--video', '-v', type=int, default=0)

    args = parser.parse_args()

    return args


def logit_loss(input, target):
    # input -> [batch_size, n_classes]
    # target -> [batch_size]
    return -input[
        torch.arange(len(target), device=input.device), target].mean()


def spatial_diff_loss(input):
    # input -> [batch_size, n_channels, height, width]
    loss = (
        ((input[:, :, 1:] - input[:, :, :-1]) ** 2).mean()
        + ((input[..., 1:] - input[..., :-1]) ** 2).mean()
    )

    return loss


def build_mask(args):
    assert os.path.isdir(args.src)
    img_paths = pathlib.Path(args.src).rglob('*.png')

    mask = []
    for path in sorted(img_paths):
        img = plt.imread(path)
        img = cv2.resize(
            img[..., 0], (args.res, args.res), interpolation=cv2.INTER_AREA)
        img = img > 0.5
        mask.append(img)

    mask = np.stack(mask)
    mask = mask.repeat(repeats=args.rep, axis=0)[:, None, ...]

    return mask


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    cnn = timm.create_model('efficientnet_b0', pretrained=True)
    resize_layer = nn.Upsample(
        size=cnn.default_cfg['input_size'][1:], mode='bicubic')

    model = nn.Sequential(resize_layer, cnn)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model


def train_step(input, target, mask, pad, model, optim, loss_fn, args):
    input = torch.where(mask, input, pad)
    output = model(input)
    class_loss = loss_fn(input=output, target=target)
    aux_loss = spatial_diff_loss(input)
    loss = class_loss + args.aux * aux_loss

    model.zero_grad()
    optim.zero_grad()
    loss.backward()
    optim.step()
    class_loss = class_loss.detach().item()
    aux_loss = aux_loss.detach().item()

    return class_loss, aux_loss


def main():

    optim_dict = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'rmsprop': torch.optim.RMSprop,
    }

    loss_fn_dict = {
        'ce': F.cross_entropy,
        'logit': logit_loss,
    }

    args = parse_args()

    device = get_device()
    model = build_model().to(device)

    mask = build_mask(args)
    mask = torch.tensor(mask, device=device)
    # mask -> [batch_size, 1, height, width]

    target = np.random.choice(
        model[-1].default_cfg['num_classes'], size=mask.shape[0])
    target = torch.tensor(target, device=device).long()
    # target -> [batch_size]

    input = torch.rand(
        size=[mask.shape[0], 3, args.res, args.res], device=device).float()

    mean, std = model[-1].default_cfg['mean'], model[-1].default_cfg['std']
    mean, std = torch.tensor(mean).to(input), torch.tensor(std).to(input)
    mean, std = mean[None, :, None, None], std[None, :, None, None]
    input = (input - mean) / std

    input = nn.Parameter(input)
    # input -> [batch_size, 3, height, width]

    optim = optim_dict[args.optim](
        [input], lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = loss_fn_dict[args.loss]
    video = []

    pad = torch.zeros(1).to(input)
    train_fn = functools.partial(train_step, pad=pad)

    pbar = tqdm.tqdm(range(args.n_iter))
    for step in pbar:
        class_loss, aux_loss = train_fn(
            input=input, target=target, mask=mask, model=model,
            optim=optim, loss_fn=loss_fn, args=args,
        )

        if args.video and not step % args.video:
            video.append(input.detach())

        pbar.set_description(f'Loss: {class_loss:.2f}, aux: {aux_loss:.2f}')

    os.makedirs(args.tgt, exist_ok=True)

    input = input.detach() * std + mean
    input = torch.where(mask, input, pad).permute(0, 2, 3, 1).cpu().numpy()
    input = np.clip(input, a_min=0, a_max=1)

    for idx, img in enumerate(input):
        #plt.figure(figsize=(12, 12))
        #plt.hist(img.flatten(), bins=100)
        # plt.show()

        path = os.path.join(args.tgt, f'{idx}.png')
        plt.imsave(path, img)


if __name__ == '__main__':
    main()
