#!/usr/bin/env python3
"""
seg_edge_stability.py

计算并可视化一组图像的分割边缘抖动指标。

用法示例:
  python seg_edge_stability.py --images-dir ./capture_data --masks-dir ./masks --output-dir ./stability_out
如果未提供 --masks-dir，脚本会尝试基于阈值自动生成掩码。
"""
import os
import sys
import argparse
from typing import List, Tuple
import csv
import shutil

import numpy as np
import cv2


def list_images(folder: str) -> List[str]:
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
    return files


def load_mask_for_image(img_path: str, masks_dir: str) -> np.ndarray:
    base = os.path.splitext(os.path.basename(img_path))[0]
    for ext in ('.png', '.jpg', '.bmp', '.tif'):
        p = os.path.join(masks_dir, base + ext)
        if os.path.exists(p):
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise RuntimeError(f'无法读取掩码: {p}')
            _, mbin = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
            return mbin
    raise FileNotFoundError(f'未找到对应掩码 for {img_path} in {masks_dir}')


def auto_mask_from_image(img: np.ndarray, blur=5, min_area=1000) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (blur, blur), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # keep largest connected component(s)
    cnts, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(th)
    for c in cnts:
        a = cv2.contourArea(c)
        if a >= min_area:
            cv2.drawContours(mask, [c], -1, 255, -1)
    if mask.sum() == 0:
        # fallback: use raw threshold
        mask = th
    return mask


def edges_from_mask(mask: np.ndarray) -> np.ndarray:
    # Canny on mask works well for boundaries
    e = cv2.Canny(mask, 50, 150)
    return e


def compute_baseline_mask(masks: List[np.ndarray]) -> np.ndarray:
    # median (majority) mask
    stack = np.stack([(m > 127).astype(np.uint8) for m in masks], axis=0)
    vote = np.sum(stack, axis=0)
    baseline = (vote >= (stack.shape[0] // 2 + 1)).astype(np.uint8) * 255
    return baseline


def boundary_distance_metrics(baseline_edges: np.ndarray, edges: np.ndarray) -> Tuple[float, float, float]:
    # distance from each edge pixel in edges to nearest baseline edge
    if baseline_edges.max() == 0 or edges.max() == 0:
        return float('nan'), float('nan'), float('nan')
    # distance transform requires 8-bit inverse: zeros are background
    inv = cv2.bitwise_not((baseline_edges > 0).astype('uint8') * 255)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    pts = np.transpose(np.nonzero(edges > 0))  # (y,x)
    if pts.shape[0] == 0:
        return float('nan'), float('nan'), float('nan')
    dists = dt[pts[:,0], pts[:,1]]
    return float(np.mean(dists)), float(np.median(dists)), float(np.max(dists))


def boundary_f1(baseline_edges: np.ndarray, edges: np.ndarray, tol=3) -> float:
    if baseline_edges.max() == 0 or edges.max() == 0:
        return float('nan')
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol*2+1, tol*2+1))
    b_d = cv2.dilate(baseline_edges, k)
    e_d = cv2.dilate(edges, k)
    match_e = np.count_nonzero((edges > 0) & (b_d > 0))
    match_b = np.count_nonzero((baseline_edges > 0) & (e_d > 0))
    prec = match_e / (np.count_nonzero(edges>0) + 1e-9)
    rec = match_b / (np.count_nonzero(baseline_edges>0) + 1e-9)
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def hausdorff_distance(a_edges: np.ndarray, b_edges: np.ndarray) -> float:
    # approximate Hausdorff by max of directed distances using distance transforms
    if a_edges.max() == 0 or b_edges.max() == 0:
        return float('nan')
    inv_b = cv2.bitwise_not((b_edges > 0).astype('uint8') * 255)
    dt_b = cv2.distanceTransform(inv_b, cv2.DIST_L2, 5)
    pts_a = np.transpose(np.nonzero(a_edges > 0))
    if pts_a.shape[0] == 0:
        return float('nan')
    d1 = np.max(dt_b[pts_a[:,0], pts_a[:,1]])
    inv_a = cv2.bitwise_not((a_edges > 0).astype('uint8') * 255)
    dt_a = cv2.distanceTransform(inv_a, cv2.DIST_L2, 5)
    pts_b = np.transpose(np.nonzero(b_edges > 0))
    if pts_b.shape[0] == 0:
        return float('nan')
    d2 = np.max(dt_a[pts_b[:,0], pts_b[:,1]])
    return float(max(d1, d2))


def overlay_edges(image: np.ndarray, baseline_edges: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = image.copy()
    # baseline edges in green, frame edges in red
    out[baseline_edges>0] = (0, 255, 0)
    # combine with red where frame edges exist (keeps red dominant)
    red_mask = edges > 0
    out[red_mask] = (0, 0, 255)
    # blend slightly for context
    blended = cv2.addWeighted(image, 0.6, out, 0.4, 0)
    return blended


def run(args):
    images = list_images(args.images_dir)
    if len(images) == 0:
        print('No images found in', args.images_dir)
        return
    os.makedirs(args.output_dir, exist_ok=True)
    masks = []
    generated_masks_dir = os.path.join(args.output_dir, 'masks')
    os.makedirs(generated_masks_dir, exist_ok=True)

    print(f'Found {len(images)} images, processing...')

    for p in images:
        img = cv2.imread(p)
        if img is None:
            raise RuntimeError('无法读取图像: ' + p)
        if args.masks_dir:
            m = load_mask_for_image(p, args.masks_dir)
        else:
            m = auto_mask_from_image(img, blur=args.blur, min_area=args.min_area)
        masks.append(m)
        # save mask for reference
        base = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(generated_masks_dir, base + '.png'), m)

    baseline_mask = compute_baseline_mask(masks)
    baseline_edges = edges_from_mask(baseline_mask)

    csv_path = os.path.join(args.output_dir, 'edge_stability.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'iou', 'mask_area', 'edge_px', 'mean_bd', 'median_bd', 'max_bd', 'bf1', 'hausdorff'])
        for img_path, mask in zip(images, masks):
            img = cv2.imread(img_path)
            edges = edges_from_mask(mask)
            # IoU
            inter = np.count_nonzero((baseline_mask>0) & (mask>0))
            union = np.count_nonzero((baseline_mask>0) | (mask>0))
            iou = inter / (union + 1e-9)
            mask_area = np.count_nonzero(mask>0)
            edge_px = np.count_nonzero(edges>0)
            mean_bd, median_bd, max_bd = boundary_distance_metrics(baseline_edges, edges)
            bf1 = boundary_f1(baseline_edges, edges, tol=args.tolerance)
            hd = hausdorff_distance(baseline_edges, edges)
            writer.writerow([os.path.basename(img_path), f'{iou:.6f}', int(mask_area), int(edge_px), f'{mean_bd:.3f}', f'{median_bd:.3f}', f'{max_bd:.3f}', f'{bf1:.3f}', f'{hd:.3f}'])
            # overlay
            ov = overlay_edges(img, baseline_edges, edges)
            outp = os.path.join(args.output_dir, os.path.basename(img_path))
            cv2.imwrite(outp, ov)

    print('Finished. Results in', args.output_dir)


def parse_args():
    p = argparse.ArgumentParser(description='Compute segmentation edge stability metrics')
    p.add_argument('--images-dir', required=True, help='Directory with input images')
    p.add_argument('--masks-dir', help='Optional directory with binary masks matching images')
    p.add_argument('--output-dir', default='./stability_out', help='Output directory')
    p.add_argument('--blur', type=int, default=5, help='Blur kernel for auto mask')
    p.add_argument('--min-area', type=int, default=500, help='Min area for objects when auto-masking')
    p.add_argument('--tolerance', type=int, default=3, help='Boundary tolerance in pixels for BF1')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
