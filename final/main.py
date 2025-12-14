# main.py
import argparse
import numpy as np
from PIL import Image
import os

from quantization import dequantize
from idct import idct2d, idct_two_1d
from ycbcr import ycbcr_to_rgb_formula, ycbcr_to_rgb_table
from metrics import Timer, compute_metrics


def decode_once(args, Y, Cb, Cr):
    block = Y[:8, :8] - 128
    block = dequantize(block, args.qtable)

    if args.idct == "2d":
        Y_rec = idct2d(block) + 128
    elif args.idct == "two1d":
        Y_rec = idct_two_1d(block) + 128
    else:
        raise ValueError("Unknown IDCT")

    if args.ycbcr == "formula":
        rgb = ycbcr_to_rgb_formula(Y_rec, Cb[:8, :8], Cr[:8, :8])
    else:
        rgb = ycbcr_to_rgb_table(Y_rec, Cb[:8, :8], Cr[:8, :8])

    return rgb


def run_experiment(args):
    gt = np.array(Image.open(args.png).convert("RGB"), dtype=np.float32)
    jpg = Image.open(args.jpg).convert("YCbCr")
    Y, Cb, Cr = [np.array(c, dtype=np.float32) for c in jpg.split()]

    times = []
    rgb = None

    with Timer() as total_timer:
        for i in range(10):
            with Timer() as t:
                rgb = decode_once(args, Y, Cb, Cr)
            times.append(t.elapsed)

    # Metrics (use last output)
    psnr, ssim = compute_metrics(gt[:8, :8], rgb)

    # Save result image
    os.makedirs(args.out_img_dir, exist_ok=True)
    out_img_path = os.path.join(
        args.out_img_dir,
        f"{args.ycbcr}_{args.idct}_Q{args.qtable}.png"
    )
    Image.fromarray(rgb.astype(np.uint8)).save(out_img_path)

    # Stats
    time_mean = np.mean(times)
    time_std = np.std(times)

    # Print (for bash parsing)
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
    parser.add_argument("--ycbcr", choices=["formula", "table"], required=True)
    parser.add_argument("--idct", choices=["2d", "two1d"], required=True)
    parser.add_argument("--qtable", type=int, choices=[1, 2], required=True)
    parser.add_argument("--out_img_dir", required=True)
    args = parser.parse_args()

    run_experiment(args)
