# main.py
import argparse
import numpy as np
from PIL import Image
import os

from quantization import dequantize_blocks
from idct import idct_blocks
from ycbcr import ycbcr_to_rgb_formula, ycbcr_to_rgb_table
from metrics import Timer, compute_metrics


def pad_to_multiple_of_8(img2d: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = img2d.shape
    ph = (8 - (h % 8)) % 8
    pw = (8 - (w % 8)) % 8
    if ph == 0 and pw == 0:
        return img2d, h, w
    padded = np.pad(img2d, ((0, ph), (0, pw)), mode="edge")
    return padded, h, w

def to_blocks(img2d: np.ndarray) -> np.ndarray:
    """
    img2d: (H, W)
    return blocks: (bh, bw, 8, 8)
    """
    H, W = img2d.shape
    bh, bw = H // 8, W // 8
    return img2d.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3).astype(np.float32)

def from_blocks(blocks: np.ndarray) -> np.ndarray:
    """
    blocks: (bh, bw, 8, 8)
    return img2d: (H, W)
    """
    bh, bw, _, _ = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(bh * 8, bw * 8)

def decode_full_image(args, Y, Cb, Cr):
    # pad each channel to multiple of 8
    Yp, oh, ow = pad_to_multiple_of_8(Y)
    Cbp, _, _ = pad_to_multiple_of_8(Cb)
    Crp, _, _ = pad_to_multiple_of_8(Cr)

    # split blocks
    Yb = to_blocks(Yp) - 128.0

    # dequantize (NOTE: this is "JPEG-like": we don't have real DCT coeffs from bitstream)
    Yb = dequantize_blocks(Yb, args.qtable)

    # IDCT blocks
    Ysp = idct_blocks(Yb, args.idct) + 128.0
    Yrec = from_blocks(Ysp)[:oh, :ow]

    # use original Cb/Cr (pixel-domain) for YCbCr->RGB (course-friendly ablation)
    if args.ycbcr == "formula":
        rgb = ycbcr_to_rgb_formula(Yrec, Cb[:oh, :ow], Cr[:oh, :ow])
    else:
        rgb = ycbcr_to_rgb_table(Yrec, Cb[:oh, :ow], Cr[:oh, :ow])

    return rgb

def run_experiment(args):
    gt = np.array(Image.open(args.png).convert("RGB"), dtype=np.float32)
    jpg = Image.open(args.jpg).convert("YCbCr")
    Y, Cb, Cr = [np.array(c, dtype=np.float32) for c in jpg.split()]

    times = []
    rgb = None

    with Timer() as total_timer:
        for _ in range(10):
            with Timer() as t:
                rgb = decode_full_image(args, Y, Cb, Cr)
            times.append(t.elapsed)

    psnr, ssim = compute_metrics(gt, rgb)

    os.makedirs(args.out_img_dir, exist_ok=True)
    out_img_path = os.path.join(
        args.out_img_dir,
        f"{args.ycbcr}_{args.idct}_Q{args.qtable}.png"
    )
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(out_img_path)

    time_mean = float(np.mean(times))
    time_std = float(np.std(times))

    print("==== Experiment Result ====")
    print(f"YCbCr        : {args.ycbcr}")
    print(f"IDCT         : {args.idct}")
    print(f"Q-Table      : {args.qtable}")
    print(f"Run times    : {','.join(f'{t:.6f}' for t in times)}")
    print(f"Time mean    : {time_mean:.6f}")
    print(f"Time std     : {time_std:.6f}")
    print(f"Time total   : {total_timer.elapsed:.6f}")
    print(f"PSNR         : {psnr:.2f}")
    print(f"SSIM         : {ssim:.4f}")
    print(f"Output image : {out_img_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", required=True)
    parser.add_argument("--jpg", required=True)

    # 2 methods
    parser.add_argument("--ycbcr", choices=["formula", "table"], required=True)

    # 3 methods -> total 12 configs
    parser.add_argument("--idct", choices=["2d", "two1d", "blocked"], required=True)

    # 2 methods
    parser.add_argument("--qtable", type=int, choices=[1, 2], required=True)

    parser.add_argument("--out_img_dir", required=True)
    args = parser.parse_args()

    run_experiment(args)
