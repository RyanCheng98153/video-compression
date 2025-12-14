# main.py
import argparse
import numpy as np
from PIL import Image
import os

from jpeg_decoder import decode_baseline_huffman
from idct import idct_blocks
from ycbcr import ycbcr_to_rgb_formula, ycbcr_to_rgb_table
from metrics import Timer, compute_metrics


def from_blocks(blocks: np.ndarray) -> np.ndarray:
    # blocks: (bh, bw, 8, 8) -> (H, W) padded
    bh, bw, _, _ = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(bh * 8, bw * 8)


def apply_qtable_per_component(blocks: np.ndarray, q: np.ndarray) -> np.ndarray:
    # blocks: (bh, bw, 8, 8), q: (8, 8)
    return blocks * q.astype(np.float32)[None, None, :, :]


def decode_full_image(args, jpeg_bytes: bytes):
    # 1) Parse + Huffman decode -> quantized DCT coeff blocks
    Yb_q, Cbb_q, Crb_q, meta = decode_baseline_huffman(
        jpeg_bytes,
        zigzag_on=(args.zigzag == "on")
    )

    H = meta["height"]
    W = meta["width"]
    qtables = meta["qtables"]
    comps = meta["components"]
    sos = meta["sos_specs"]  # scan order

    # Map SOS order -> cid for Y/Cb/Cr returned by decoder
    cid_Y, cid_Cb, cid_Cr = [s.cid for s in sos]

    # 2) Use REAL JPEG DQT tables (per component tq id)
    qY = qtables[comps[cid_Y].tq]
    qC = qtables[comps[cid_Cb].tq]  # Cb and Cr usually share the same table id
    qR = qtables[comps[cid_Cr].tq]

    # 3) Dequantize with correct table
    Yb = apply_qtable_per_component(Yb_q, qY)
    Cbb = apply_qtable_per_component(Cbb_q, qC)
    Crb = apply_qtable_per_component(Crb_q, qR)

    # 4) IDCT ablation (spatial blocks) + level shift
    Ysp = idct_blocks(Yb, args.idct) + 128.0
    Cbsp = idct_blocks(Cbb, args.idct) + 128.0
    Crsp = idct_blocks(Crb, args.idct) + 128.0

    # 5) Stitch blocks -> full planes, crop to (H,W)
    Y = from_blocks(Ysp)[:H, :W]
    Cb = from_blocks(Cbsp)[:H, :W]
    Cr = from_blocks(Crsp)[:H, :W]

    # 6) YCbCr -> RGB ablation
    if args.ycbcr == "formula":
        rgb = ycbcr_to_rgb_formula(Y, Cb, Cr)
    else:
        rgb = ycbcr_to_rgb_table(Y, Cb, Cr)

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

    # Save output image
    os.makedirs(args.out_img_dir, exist_ok=True)
    out_img_path = os.path.join(
        args.out_img_dir,
        f"{args.ycbcr}_{args.idct}_zigzag_{args.zigzag}.png"
    )
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(out_img_path)

    time_mean = float(np.mean(times))
    time_std = float(np.std(times))

    print("==== Experiment Result ====")
    print(f"YCbCr        : {args.ycbcr}")
    print(f"IDCT         : {args.idct}")
    print(f"ZigZag       : {args.zigzag}")
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

    # Ablation 1: 2 methods
    parser.add_argument("--ycbcr", choices=["formula", "table"], required=True)

    # Ablation 2: 3 methods
    parser.add_argument("--idct", choices=["2d", "two1d", "blocked"], required=True)

    # Ablation 3: ZigZag on/off
    parser.add_argument("--zigzag", choices=["on", "off"], default="on")

    parser.add_argument("--out_img_dir", required=True)
    args = parser.parse_args()

    run_experiment(args)
