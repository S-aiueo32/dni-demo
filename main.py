from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from google_drive_downloader import GoogleDriveDownloader as gdd
from PIL import Image, ImageDraw

from model import MSRResNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def download_weights(save_dir='./weights'):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    gdd.download_file_from_google_drive(
        file_id='1NlwvfiDk3UgNHz63GldPOXBXxKw4ScXq',
        dest_path=save_dir / 'MSRResNetx4.pth'
    )
    gdd.download_file_from_google_drive(
        file_id='1c0YNygNMfTLynR-C3y7nsZgaWbczbW5j',
        dest_path=save_dir / 'MSRGANx4.pth'
    )


def convert_weights(old_state_dict):
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        if 'conv_first' in new_key:
            new_key = new_key.replace('conv_first', 'head.0')
        if 'recon_trunk' in new_key:
            new_key = new_key.replace('recon_trunk', 'body')
        if '.conv1.weight' in new_key:
            new_key = new_key.replace('.conv1.weight', '.body.0.weight')
        if '.conv1.bias' in new_key:
            new_key = new_key.replace('.conv1.bias', '.body.0.bias')
        if '.conv2.weight' in new_key:
            new_key = new_key.replace('.conv2.weight', '.body.2.weight')
        if '.conv2.bias' in new_key:
            new_key = new_key.replace('.conv2.bias', '.body.2.bias')
        if 'upconv1' in new_key:
            new_key = new_key.replace('upconv1', 'tail.0')
        if 'upconv2' in new_key:
            new_key = new_key.replace('upconv2', 'tail.3')
        if 'HRconv' in new_key:
            new_key = new_key.replace('HRconv', 'tail.6')
        if 'conv_last' in new_key:
            new_key = new_key.replace('conv_last', 'tail.8')
        new_state_dict[new_key] = val
    return new_state_dict


def interpolate_network(netA, netB, alpha):
    net_interp = OrderedDict()
    for k in netA.keys():
        v_A, v_B = netA[k], netB[k]
        net_interp[k.replace('module.', '')] = alpha * v_A + (1 - alpha) * v_B
    return net_interp


def load_image(path):
    hr_img = Image.open(path).convert('RGB')
    hr_img = hr_img.resize((x + x % 4 for x in hr_img.size), Image.BICUBIC)
    lr_img = hr_img.resize((x // 4 for x in hr_img.size), Image.BICUBIC)

    hr_tensor = TF.to_tensor(hr_img).unsqueeze(0).to(device)
    lr_tensor = TF.to_tensor(lr_img).unsqueeze(0).to(device)

    return hr_tensor, lr_tensor


def main(args):
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)

    download_weights()

    model = MSRResNet().to(device)

    netA = convert_weights(torch.load('./weights/MSRGANx4.pth'))
    netB = convert_weights(torch.load('./weights/MSRResNetx4.pth'))

    imgs = []
    for alpha in np.arange(0.0, 1.1, 0.1):
        net_interp = interpolate_network(netA, netB, alpha)
        model.load_state_dict(net_interp)

        with torch.no_grad():
            y, x = load_image(args.input)
            y_hat = model(x).clamp(0, 1)

        out_name = f'{Path(args.input).stem}_alpha{alpha:.1f}.png'
        save_image(y_hat.squeeze(0), out_dir / out_name)

        img = Image.open(out_dir / out_name)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), f'alpha = {alpha:.1f}')
        imgs.append(img)

    imgs[0].save(
        out_dir / f'{Path(args.input).stem}.gif',
        save_all=True,
        append_images=imgs[1:],
        duration=500,
        loop=0
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='./data/lenna.png')
    args = parser.parse_args()

    main(args)
