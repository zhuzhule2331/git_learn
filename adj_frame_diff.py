#!/usr/bin/env python3
"""
adj_frame_diff.py

计算一组图像中相邻帧之间的像素级差异统计（mean abs diff, max, MSE, PSNR, SSIM），并保存差异图与 CSV 报表。

用法示例:
  python adj_frame_diff.py --images-dir ./capture_data --output-dir ./adj_diff_out --save-diffs
"""
import os
import argparse
import csv
from typing import List

import numpy as np
import cv2


def list_images(folder: str) -> List[str]:
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
    return files


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr_val(mse_v: float, max_pixel=255.0) -> float:
    if mse_v == 0:
        return float('inf')
    return 20.0 * np.log10(max_pixel / np.sqrt(mse_v))


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    # simplified SSIM implementation for grayscale images
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T
    mu1 = cv2.filter2D(a, -1, window)
    mu2 = cv2.filter2D(b, -1, window)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(a * a, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(b * b, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(a * b, -1, window) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def analyze_pair(img1, img2):
    if img1.shape != img2.shape:
        raise RuntimeError('Image size mismatch')
    absdiff = cv2.absdiff(img1, img2)
    mean_abs = float(np.mean(absdiff))
    max_abs = int(np.max(absdiff))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mse_v = mse(gray1, gray2)
    psnr_v = psnr_val(mse_v)
    ssim_v = ssim_gray(gray1, gray2)
    return {
        'mean_abs': mean_abs,
        'max_abs': max_abs,
        'mse': mse_v,
        'psnr': psnr_v,
        'ssim': ssim_v,
        'absdiff': absdiff,
    }


def run(args):
    images = list_images(args.images_dir)
    if len(images) < 2:
        print('Need at least 2 images in', args.images_dir)
        return
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'adj_frame_diff.csv')
    with open(csv_path, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['pair_index', 'img_prev', 'img_next', 'mean_abs', 'max_abs', 'mse', 'psnr', 'ssim'])
        for i in range(1, len(images)):
            p0 = images[i-1]
            p1 = images[i]
            im0 = cv2.imread(p0)
            im1 = cv2.imread(p1)
            if im0 is None or im1 is None:
                print('Failed reading', p0, p1)
                continue
            if im0.shape != im1.shape:
                print('Skipping pair with shape mismatch:', p0, p1)
                continue
            res = analyze_pair(im0, im1)
            writer.writerow([i-1, os.path.basename(p0), os.path.basename(p1), f'{res["mean_abs"]:.3f}', res['max_abs'], f'{res["mse"]:.3f}', f'{res["psnr"]:.3f}', f'{res["ssim"]:.6f}'])
            if args.save_diffs:
                outp = os.path.join(args.output_dir, f'diff_{i-1:03d}_{os.path.basename(p0)}')
                # write normalized absdiff for visualization
                ad = res['absdiff']
                # scale to 0-255 if multi-channel
                cv2.imwrite(outp, ad)
    print('Done. CSV:', csv_path)


def parse_args():
    p = argparse.ArgumentParser(description='Adjacent frame differences')
    p.add_argument('--images-dir', required=True, help='Directory with images')
    p.add_argument('--output-dir', default='./adj_diff_out', help='Output directory')
    p.add_argument('--save-diffs', action='store_true', help='Save absdiff images')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
