# main.py
import argparse
import numpy as np
from PIL import Image

from quantization import dequantize
from idct import idct2d, idct_two_1d
from ycbcr import ycbcr_to_rgb_formula, ycbcr_to_rgb_table
from metrics import Timer, compute_metrics

def run_experiment(args):
    gt = np.array(Image.open(args.png).convert("RGB"), dtype=np.float32)
    jpg = Image.open(args.jpg).convert("YCbCr")
    Y, Cb, Cr = [np.array(c, dtype=np.float32) for c in jpg.split()]

    with Timer() as t:
        # fake DCT coeffs for demo (replace with real coeffs)
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

    psnr, ssim = compute_metrics(gt[:8, :8], rgb)

    print("==== Experiment Result ====")
    print(f"YCbCr     : {args.ycbcr}")
    print(f"IDCT      : {args.idct}")
    print(f"Q-Table   : {args.qtable}")
    print(f"Time (s)  : {t.elapsed:.6f}")
    print(f"PSNR      : {psnr:.2f}")
    print(f"SSIM      : {ssim:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", required=True)
    parser.add_argument("--jpg", required=True)
    parser.add_argument("--ycbcr", choices=["formula", "table"], default="formula")
    parser.add_argument("--idct", choices=["2d", "two1d"], default="2d")
    parser.add_argument("--qtable", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()

    run_experiment(args)
