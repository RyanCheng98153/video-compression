import cv2
import numpy as np
import argparse
import os
import time

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def ensure_gray_u8(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)

def crop_to_block_multiple(img, blk):
    h, w = img.shape
    H = h - (h % blk)
    W = w - (w % blk)
    return img[:H, :W]

def psnr(gt_u8, pred_u8):
    gt = gt_u8.astype(np.float32)
    pr = pred_u8.astype(np.float32)
    mse = np.mean((gt - pr) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)

def save_outputs(prefix, out_dir, recon_u8, residual_i16):
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(f"{out_dir}/{prefix}_recon.png", recon_u8)

    resid_vis = np.clip(residual_i16 + 128, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{out_dir}/{prefix}_residual_vis.png", resid_vis)

    np.save(f"{out_dir}/{prefix}_residual_raw.npy", residual_i16.astype(np.int16))

def pad_for_search(img, pad):
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

# ------------------------------------------------------------
# Full Search Motion Estimation
# ------------------------------------------------------------
def full_search_ME(ref, cur, blk=8, search_range=8):
    H, W = cur.shape
    pr = pad_for_search(ref, search_range)
    recon = np.zeros_like(cur, dtype=np.uint8)

    for by in range(0, H, blk):
        for bx in range(0, W, blk):
            best_cost = 1e18
            best_dy = 0
            best_dx = 0
            cur_blk = cur[by:by+blk, bx:bx+blk].astype(np.int16)

            ry = by + search_range
            rx = bx + search_range

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ref_blk = pr[ry + dy: ry + dy + blk, rx + dx: rx + dx + blk].astype(np.int16)
                    sad = np.abs(cur_blk - ref_blk).sum()
                    if sad < best_cost:
                        best_cost = sad
                        best_dy, best_dx = dy, dx

            recon[by:by+blk, bx:bx+blk] = pr[ry + best_dy: ry + best_dy + blk,
                                             rx + best_dx: rx + best_dx + blk]

    return recon

# ------------------------------------------------------------
# Three-Step Search Motion Estimation
# ------------------------------------------------------------
def three_step_search_ME(ref, cur, blk=8, search_range=8):
    H, W = cur.shape
    pr = pad_for_search(ref, search_range)
    recon = np.zeros_like(cur, dtype=np.uint8)

    if search_range < 1:
        step0 = 1
    else:
        step0 = 1 << int(np.floor(np.log2(search_range)))

    def sad_block(a, b):
        return np.abs(a.astype(np.int16) - b.astype(np.int16)).sum()

    for by in range(0, H, blk):
        for bx in range(0, W, blk):
            ry = by + search_range
            rx = bx + search_range

            cur_blk = cur[by:by+blk, bx:bx+blk]

            cy, cx = 0, 0
            step = step0

            for _ in range(3):
                best_cost = sad_block(cur_blk, pr[ry+cy:ry+cy+blk, rx+cx:rx+cx+blk])
                best_dy, best_dx = cy, cx

                for dy in (-step, 0, step):
                    for dx in (-step, 0, step):
                        ny, nx = cy + dy, cx + dx
                        if abs(ny) <= search_range and abs(nx) <= search_range:
                            ref_blk = pr[ry+ny:ry+ny+blk, rx+nx:rx+nx+blk]
                            c = sad_block(cur_blk, ref_blk)
                            if c < best_cost:
                                best_cost, best_dy, best_dx = c, ny, nx

                cy, cx = best_dy, best_dx
                step = max(1, step // 2)

            recon[by:by+blk, bx:bx+blk] = pr[ry+cy:ry+cy+blk, rx+cx:rx+cx+blk]

    return recon

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    # Output directories (fixed)
    full_dir = "figures/full"
    tss_dir = "figures/tss"
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(tss_dir, exist_ok=True)

    ref = ensure_gray_u8(cv2.imread(args.ref_img, cv2.IMREAD_UNCHANGED))
    cur = ensure_gray_u8(cv2.imread(args.cur_img, cv2.IMREAD_UNCHANGED))

    H = min(ref.shape[0], cur.shape[0])
    W = min(ref.shape[1], cur.shape[1])
    ref = crop_to_block_multiple(ref[:H, :W], 8)
    cur = crop_to_block_multiple(cur[:H, :W], 8)

    ranges = [8, 16, 32]

    print("Running full search...")
    for R in ranges:
        t0 = time.perf_counter()
        recon = full_search_ME(ref, cur, blk=8, search_range=R)
        dt = time.perf_counter() - t0
        p = psnr(cur, recon)
        residual = cur.astype(np.int16) - recon.astype(np.int16)

        print(f"[Full] range=±{R:>2} | PSNR={p:6.2f} dB | time={dt:6.3f} s")
        save_outputs(f"full_r{R}", full_dir, recon, residual)

    print("Running TSS...")
    for R in ranges:
        t0 = time.perf_counter()
        recon = three_step_search_ME(ref, cur, blk=8, search_range=R)
        dt = time.perf_counter() - t0
        p = psnr(cur, recon)
        residual = cur.astype(np.int16) - recon.astype(np.int16)

        print(f"[TSS ] range=±{R:>2} | PSNR={p:6.2f} dB | time={dt:6.3f} s")
        save_outputs(f"tss_r{R}", tss_dir, recon, residual)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_img", required=True, help="Reference image path")
    parser.add_argument("--cur_img", required=True, help="Current image path")
    args = parser.parse_args()

    main(args)
