import cv2
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import os

# ================================
# Quantization Tables 
# ================================
qtable1 = np.array([
    [10, 7, 6, 10, 14, 24, 31, 37],
    [7, 7, 8, 11, 16, 35, 36, 33],
    [8, 8, 10, 14, 24, 34, 41, 34],
    [8, 10, 13, 17, 31, 52, 48, 37],
    [11, 13, 22, 34, 41, 65, 62, 46],
    [14, 21, 33, 38, 49, 62, 68, 55],
    [29, 38, 47, 52, 62, 73, 72, 61],
    [43, 55, 57, 59, 67, 60, 62, 59]
], dtype=np.float32)

qtable2 = np.array([
    [10, 11, 14, 28, 59, 59, 59, 59],
    [11, 13, 16, 40, 59, 59, 59, 59],
    [14, 16, 34, 59, 59, 59, 59, 59],
    [28, 40, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59]
], dtype=np.float32)

# ================================
# Zigzag
# ================================
def zigzag_indices(n=8):
    idx = []
    for s in range(2*n - 1):
        if s % 2 == 0:
            r = min(s, n-1)
            c = s - r
            while r >= 0 and c < n:
                idx.append((r,c))
                r -= 1
                c += 1
        else:
            c = min(s, n-1)
            r = s - c
            while c >= 0 and r < n:
                idx.append((r,c))
                r += 1
                c -= 1
    return idx

ZIGZAG = zigzag_indices(8)

# ================================
# Block operations
# ================================
def pad_image_to_blocksize(img, block_size=8):
    h, w = img.shape
    ph = ceil(h/block_size)*block_size
    pw = ceil(w/block_size)*block_size
    pad_h, pad_w = ph-h, pw-w
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                cv2.BORDER_CONSTANT, value=0)
    return padded, h, w, ph, pw

def split_blocks(img, block_size=8):
    padded, orig_h, orig_w, full_h, full_w = pad_image_to_blocksize(img)
    blocks = []
    for y in range(0, full_h, block_size):
        for x in range(0, full_w, block_size):
            blocks.append(padded[y:y+8, x:x+8].astype(np.float32))
    return np.array(blocks), orig_h, orig_w, full_h, full_w

def merge_blocks(blocks, full_h, full_w, block_size=8, orig_h=None, orig_w=None):
    out = np.zeros((full_h, full_w), dtype=np.float32)
    idx = 0
    for y in range(0, full_h, block_size):
        for x in range(0, full_w, block_size):
            out[y:y+8, x:x+8] = blocks[idx]
            idx += 1
    if orig_h is not None:
        out = out[:orig_h, :orig_w]
    return out


# ================================
# DCT / IDCT
# ================================
def blocks_dct(blocks):
    out = np.empty_like(blocks)
    for i, b in enumerate(blocks):
        out[i] = cv2.dct(b - 128.0)
    return out

def blocks_idct(blocks):
    out = np.empty_like(blocks)
    for i, b in enumerate(blocks):
        ib = cv2.idct(b) + 128.0
        out[i] = np.clip(ib, 0, 255)
    return out

# ================================
# Quant / Dequant
# ================================
def quantize_blocks(dct_blocks, qtable):
    q = qtable.astype(np.float32)
    out = np.empty_like(dct_blocks)
    for i in range(len(dct_blocks)):
        out[i] = np.round(dct_blocks[i] / q)
    return out.astype(np.int32)

def dequantize_blocks(quant_blocks, qtable):
    q = qtable.astype(np.float32)
    out = np.empty_like(quant_blocks, dtype=np.float32)
    for i in range(len(quant_blocks)):
        out[i] = quant_blocks[i] * q
    return out

# ================================
# Zigzag + RLE
# ================================
def block_to_zigzag_vector(block):
    v = np.zeros(64, dtype=block.dtype)
    for i, (r,c) in enumerate(ZIGZAG):
        v[i] = block[r,c]
    return v

def zigzag_vector_to_block(vec):
    out = np.zeros((8,8), dtype=vec.dtype)
    for i, (r,c) in enumerate(ZIGZAG):
        out[r,c] = vec[i]
    return out

def rle_encode_block(block):
    vec = block_to_zigzag_vector(block)
    tokens = []
    run = 0
    for v in vec:
        if v == 0:
            run += 1
        else:
            tokens.append((run, int(v)))
            run = 0
    tokens.append(("EOB",))
    return tokens

def rle_decode_block(tokens):
    vec = np.zeros(64, dtype=np.int32)
    idx = 0
    for t in tokens:
        if t[0] == "EOB":
            break
        run, val = t
        idx += run
        if idx < 64:
            vec[idx] = val
        idx += 1
    return zigzag_vector_to_block(vec)

# ================================
# Single-channel processing 
# ================================
def process_with_qtable(img_gray, qtable, save_prefix):
    blocks, orig_h, orig_w, full_h, full_w = split_blocks(img_gray)
    dctb = blocks_dct(blocks)
    quant = quantize_blocks(dctb, qtable)

    tokens_all = [rle_encode_block(qb) for qb in quant]

    # reconstruct
    decoded = np.array([rle_decode_block(tk) for tk in tokens_all])
    deq = dequantize_blocks(decoded, qtable)
    rec_blocks = blocks_idct(deq)
    rec_img = merge_blocks(rec_blocks, full_h, full_w, 8, orig_h, orig_w)
    rec_img = np.round(rec_img).astype(np.uint8)

    stats = {
        "num_blocks": len(blocks),
        "total_tokens": sum(len(tk) for tk in tokens_all)
    }

    return rec_img, tokens_all, stats

# ================================
# Image Estimation
# ================================

def estimate_bytes_from_tokens(all_tokens):
    """
    all_tokens: list of token lists from all color channels
    return: estimated byte size using JPEG-like RLE coding
    """
    bytes_est = 0
    for toks in all_tokens:
        for t in toks:
            if t[0] == "EOB":
                bytes_est += 1
            else:
                bytes_est += 3
    return bytes_est

def compute_psnr(img1, img2):
    """
    Compute PSNR between two images (grayscale or RGB).
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def plot_bar_comparison(values, labels, title, ylabel, save_path):
    """
    Plot a bar chart comparing two values.
    values: [v1, v2]
    labels: ["QTable1", "QTable2"]
    """
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ================================
# Color pipeline
# ================================

def process_color_image(img_color, qtable, save_prefix):
    b, g, r = cv2.split(img_color)

    rec_channels = []
    for ch, name in zip([b,g,r], ["B","G","R"]):
        rec_ch, tokens, stats = process_with_qtable(ch, qtable,
                                                    save_prefix + "_" + name)
        rec_channels.append(rec_ch)

    rec_img = cv2.merge(rec_channels)
    return rec_img

def process_color_image(img_color, qtable, save_prefix):
    b, g, r = cv2.split(img_color)

    rec_list = []
    all_tokens = [] # For byte estimation

    for ch, name in zip([b, g, r], ["B", "G", "R"]):
        rec_ch, tokens, _ = process_with_qtable(
            ch, qtable, save_prefix + "_" + name
        )
        rec_list.append(rec_ch)
        all_tokens.extend(tokens) # For byte estimation

    rec_img = cv2.merge(rec_list)
    return rec_img, all_tokens


# ================================
# Main
# ================================

import argparse

def main():
    
    parser = argparse.ArgumentParser(description="Process an image with different quantization tables.")
    parser.add_argument("--infile", default="lena.png", type=str, help="Path to the input image")
    args = parser.parse_args()
    
    img = cv2.imread(args.infile)
    if img is None:
        print(f"Cannot load {args.infile}")
        return

    os.makedirs("figures", exist_ok=True)

    # Main Process QTable1
    print("Processing COLOR image with QTable1...")
    save_prefix = "qtable1_color"
    rec_img_1, tokens_1 = process_color_image(img, qtable1, save_prefix)
    cv2.imwrite(f"figures/{save_prefix}_reconstructed.png", rec_img_1)

    # Main Process QTable2
    print("Processing COLOR image with QTable2...")
    save_prefix = "qtable2_color"
    rec_img_2, tokens_2 = process_color_image(img, qtable2, save_prefix)
    cv2.imwrite(f"figures/{save_prefix}_reconstructed.png", rec_img_2)
    
    # Estimate bytes
    bytes1 = estimate_bytes_from_tokens(tokens_1)
    bytes2 = estimate_bytes_from_tokens(tokens_2)
    
    # Calculate PSNR (RGB)
    psnr1 = compute_psnr(img, rec_img_1)
    psnr2 = compute_psnr(img, rec_img_2)

    print("\n===== QTable COMPARISON =====")
    print("QTable1 Bytes:", bytes1)
    print("QTable2 Bytes:", bytes2)
    print("QTable1 PSNR:", psnr1)
    print("QTable2 PSNR:", psnr2)
    
    # Bar plot for size comparison
    plot_bar_comparison(
        [bytes1, bytes2],
        ["QTable1", "QTable2"],
        "Encoded Size Comparison",
        "Estimated Bytes",
        "./encoded_size_bar.png"
    )

    print("Done. Output in ./figures folder.")

if __name__ == "__main__":
    main()
