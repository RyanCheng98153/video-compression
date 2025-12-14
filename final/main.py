# main.py
import argparse
import numpy as np
from PIL import Image
import os

from jpeg_decoder import decode_baseline_huffman
from quantization import dequantize_blocks
from idct import idct2d, idct_two_1d, idct_block_based
from ycbcr import ycbcr_to_rgb_formula, ycbcr_to_rgb_table
from metrics import compute_metrics, Timer


def apply_idct_per_block(blocks, idct_fn):
    """
    blocks : (bh, bw, 8, 8)
    return : (H, W)
    """
    bh, bw, _, _ = blocks.shape
    H = bh * 8
    W = bw * 8
    out = np.zeros((H, W), dtype=np.float32)

    for by in range(bh):
        for bx in range(bw):
            out[
                by*8:(by+1)*8,
                bx*8:(bx+1)*8
            ] = idct_fn(blocks[by, bx])

    return out


def decode_full_image(args, jpeg_bytes: bytes):
    # ------------------------------------------------------------
    # 1. Entropy + Huffman decode
    # ------------------------------------------------------------
    Yb_q, Cbb_q, Crb_q, meta = decode_baseline_huffman(
        jpeg_bytes,
        zigzag_on=True
    )

    H = meta["height"]
    W = meta["width"]
    qtables = meta["qtables"]
    comps = meta["components"]

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
    # 3. IDCT (ABLATION)  ✅ FIXED
    # ------------------------------------------------------------
    if args.idct == "2d":
        Y  = apply_idct_per_block(Yb,  idct2d)
        Cb = apply_idct_per_block(Cbb, idct2d)
        Cr = apply_idct_per_block(Crb, idct2d)

    elif args.idct == "two1d":
        Y  = apply_idct_per_block(Yb,  idct_two_1d)
        Cb = apply_idct_per_block(Cbb, idct_two_1d)
        Cr = apply_idct_per_block(Crb, idct_two_1d)

    elif args.idct == "block":
        Y  = idct_block_based(Yb)
        Cb = idct_block_based(Cbb)
        Cr = idct_block_based(Crb)

    else:
        raise ValueError("Unknown IDCT method")

    # ------------------------------------------------------------
    # 3.5 LEVEL SHIFT (CRITICAL FIX)
    # ------------------------------------------------------------
    Y  = Y  + 128.0
    Cb = Cb + 128.0
    Cr = Cr + 128.0

    Y  = np.clip(Y,  0, 255)
    Cb = np.clip(Cb, 0, 255)
    Cr = np.clip(Cr, 0, 255)
        
    # ------------------------------------------------------------
    # 4. YCbCr → RGB
    # ------------------------------------------------------------
    if args.ycbcr == "formula":
        rgb = ycbcr_to_rgb_formula(Y, Cb, Cr)
    else:
        rgb = ycbcr_to_rgb_table(Y, Cb, Cr)

    # Crop padding
    rgb = rgb[:H, :W]
    return rgb


def run_experiment(args):
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

    os.makedirs(args.out_img_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_img_dir,
        f"{args.ycbcr}_{args.idct}_{args.dequant}.png"
    )
    Image.fromarray(rgb.astype(np.uint8)).save(out_path)

    print("==== Experiment Result ====")
    print(f"YCbCr        : {args.ycbcr}")
    print(f"IDCT         : {args.idct}")
    print(f"Dequant      : {args.dequant}")
    print(f"Run times    : {','.join(f'{t:.6f}' for t in times)}")
    print(f"Time mean    : {np.mean(times):.6f}")
    print(f"Time std     : {np.std(times):.6f}")
    print(f"Time total   : {total_timer.elapsed:.6f}")
    print(f"PSNR         : {psnr:.2f}")
    print(f"SSIM         : {ssim:.4f}")
    print(f"Output image : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", required=True)
    parser.add_argument("--jpg", required=True)
    parser.add_argument("--ycbcr", choices=["formula", "table"], required=True)
    parser.add_argument("--idct", choices=["2d", "two1d", "block"], required=True)
    parser.add_argument("--dequant", choices=["float", "int"], required=True)
    parser.add_argument("--out_img_dir", required=True)
    args = parser.parse_args()

    run_experiment(args)
