from argparse import ArgumentParser
from collections import OrderedDict
from math import sqrt
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from google_drive_downloader import GoogleDriveDownloader as gdd
from PIL import Image, ImageDraw
from skvideo import measure

from model import MSRResNet, RRDBNet

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
    gdd.download_file_from_google_drive(
        file_id='1yWDdoslDhT7G5TmXmCHSOF1c89fSwRb8',
        dest_path=save_dir / 'RRDB_ESRGAN_x4.pth'
    )
    gdd.download_file_from_google_drive(
        file_id='13fEJuPDCN2meSgWhld6ru4L3-ZoNvGki',
        dest_path=save_dir / 'RRDB_PSNR_x4.pth'
    )


def interpolate_network(netA, netB, alpha):
    net_interp = OrderedDict()
    for k in netA.keys():
        v_A, v_B = netA[k], netB[k]
        net_interp[k.replace('module.', '')] = alpha * v_A + (1 - alpha) * v_B
    return net_interp


def tensor_to_array(tensor):
    tensor = tensor.cpu().squeeze(0).permute(1, 2, 0)
    return (tensor.numpy() * 255).astype('uint8')


def compute_metrics(prediction, target):
    prediction = tensor_to_array(prediction)
    target = tensor_to_array(target)

    prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY)
    target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

    mse, *_ = measure.mse(target, prediction)
    niqe, *_ = measure.niqe(prediction)

    return sqrt(mse), niqe


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

    if args.net == 'msrresnet':
        model = MSRResNet().to(device)
        netA = torch.load('./weights/MSRGANx4.pth')
        netB = torch.load('./weights/MSRResNetx4.pth')
    if args.net == 'rrdb':
        model = RRDBNet(3, 3, 64, 23).to(device)
        netA = torch.load('./weights/RRDB_ESRGAN_x4.pth')
        netB = torch.load('./weights/RRDB_PSNR_x4.pth')

    imgs = []
    rmse_list, niqe_list = [], []
    for alpha in np.arange(0.0, 1.1, 0.1):
        net_interp = interpolate_network(netA, netB, alpha)
        model.load_state_dict(net_interp)

        with torch.no_grad():
            y, x = load_image(args.input)
            y_hat = model(x).clamp(0, 1)

        rmse, niqe = compute_metrics(y_hat, y)
        rmse_list.append(rmse)
        niqe_list.append(niqe)

        out_name = f'{Path(args.input).stem}_alpha{alpha:.1f}.png'
        save_image(y_hat.squeeze(0), out_dir / out_name)

        img = Image.open(out_dir / out_name)
        draw = ImageDraw.Draw(img)
        draw.text(
            (0, 0), f'alpha = {alpha:.1f}\nRMSE:{rmse:.2f}, NIQE:{niqe:.2f}')
        imgs.append(img)

    imgs[0].save(
        out_dir / f'{Path(args.input).stem}.gif',
        save_all=True,
        append_images=imgs[1:],
        duration=500,
        loop=0
    )

    plt.plot(rmse_list, niqe_list, 'x-')
    plt.xlabel('RMSE')
    plt.ylabel('NIQE')
    plt.savefig(out_dir / f'{Path(args.input).stem}_graph.png')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='./data/baboon.png')
    parser.add_argument('--net', default='rrdb', choices=['msrresnet', 'rrdb'])
    args = parser.parse_args()

    main(args)
