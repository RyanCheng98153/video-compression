# main.py
import argparse
import numpy as np
from PIL import Image
import os
import time

from jpeg_decoder import decode_baseline_huffman
from quantization import dequantize_blocks
from idct import idct2d, idct_two_1d, idct_blocks
from ycbcr import ycbcr_to_rgb_formula, ycbcr_to_rgb_table
from metrics import compute_metrics, Timer


def decode_full_image(args, jpeg_bytes: bytes):
    # ------------------------------------------------------------
    # 1. Entropy + Huffman decode (real JPEG decoder)
    # ------------------------------------------------------------
    Yb_q, Cbb_q, Crb_q, meta = decode_baseline_huffman(
        jpeg_bytes,
        zigzag_on=True  # ZigZag is mandatory
    )

    H = meta["height"]
    W = meta["width"]
    qtables = meta["qtables"]
    comps = meta["components"]

    # Component → quant table
    qY  = qtables[comps[1].tq]
    qCb = qtables[comps[2].tq]
    qCr = qtables[comps[3].tq]

    # ------------------------------------------------------------
    # 2. Dequantization (ABLATION)
    # ------------------------------------------------------------
    Yb  = dequantize_blocks(Yb_q,  qY,  args.dequant)
    Cbb = dequantize_blocks(Cbb_q, qCb, args.dequant)
    Crb = dequantize_blocks(Crb_q, qCr, args.dequant)

    # ------------------------------------------------------------
    # 3. IDCT (ABLATION)
    # ------------------------------------------------------------
    if args.idct == "2d":
        Y  = idct2d(Yb)
        Cb = idct2d(Cbb)
        Cr = idct2d(Crb)

    elif args.idct == "two1d":
        Y  = idct_two_1d(Yb)
        Cb = idct_two_1d(Cbb)
        Cr = idct_two_1d(Crb)

    elif args.idct == "block":
        Y  = idct_block_based(Yb)
        Cb = idct_block_based(Cbb)
        Cr = idct_block_based(Crb)

    else:
        raise ValueError("Unknown IDCT method")

    # ------------------------------------------------------------
    # 4. YCbCr → RGB (ABLATION)
    # ------------------------------------------------------------
    if args.ycbcr == "formula":
        rgb = ycbcr_to_rgb_formula(Y, Cb, Cr)
    else:
        rgb = ycbcr_to_rgb_table(Y, Cb, Cr)

    # Crop padding
    rgb = rgb[:H, :W]
    return rgb


def run_experiment(args):
    # Ground truth
    gt = np.array(Image.open(args.png).convert("RGB"), dtype=np.float32)

    with open(args.jpg, "rb") as f:
        jpeg_bytes = f.read()

    times = []
    rgb = None

    with Timer() as total_timer:
        for _ in range(10):
            with Timer() as t:
                rgb = decode_full_image(args, jpeg_bytes)
            times.append(t.elapsed)

    psnr, ssim = compute_metrics(gt, rgb)

    # Save image
    os.makedirs(args.out_img_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_img_dir,
        f"{args.ycbcr}_{args.idct}_{args.dequant}.png"
    )
    Image.fromarray(rgb.astype(np.uint8)).save(out_path)

    # Stats
    time_mean = np.mean(times)
    time_std  = np.std(times)

    # Print for bash parsing
    print("==== Experiment Result ====")
    print(f"YCbCr        : {args.ycbcr}")
    print(f"IDCT         : {args.idct}")
    print(f"Dequant      : {args.dequant}")
    print(f"Run times    : {','.join(f'{t:.6f}' for t in times)}")
    print(f"Time mean    : {time_mean:.6f}")
    print(f"Time std     : {time_std:.6f}")
    print(f"Time total   : {total_timer.elapsed:.6f}")
    print(f"PSNR         : {psnr:.2f}")
    print(f"SSIM         : {ssim:.4f}")
    print(f"Output image : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", required=True)
    parser.add_argument("--jpg", required=True)

    # Ablation dimensions
    parser.add_argument("--ycbcr", choices=["formula", "table"], required=True)
    parser.add_argument("--idct", choices=["2d", "two1d", "block"], required=True)
    parser.add_argument("--dequant", choices=["float", "int"], required=True)

    parser.add_argument("--out_img_dir", required=True)
    args = parser.parse_args()

    run_experiment(args)
