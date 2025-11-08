import cv2
import numpy as np
import time
from math import log2, ceil

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def ensure_gray_u8(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

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
        return float('inf')
    return 10.0 * np.log10((255.0 ** 2) / mse)

def save_outputs(prefix, recon_u8, residual_i16):
    # Reconstructed frame
    cv2.imwrite(f"{prefix}_recon.png", recon_u8)
    # Residual as viewable PNG (centered at 128)
    resid_vis = np.clip(residual_i16 + 128, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{prefix}_residual_vis.png", resid_vis)
    # Residual raw values for exactness
    np.save(f"{prefix}_residual_raw.npy", residual_i16.astype(np.int16))

def pad_for_search(img, pad):
    # Replicate padding so we can index safely during search
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

# ------------------------------------------------------------
# Block matchers (SAD)
# ------------------------------------------------------------
def full_search_ME(ref, cur, blk=8, search_range=8):
    """
    ref, cur: uint8 2D arrays (H, W)
    Returns:
      recon (uint8), mv_yx (H//blk, W//blk, 2) dy, dx
    """
    H, W = cur.shape
    pr = pad_for_search(ref, search_range)
    recon = np.zeros_like(cur, dtype=np.uint8)
    mv = np.zeros((H // blk, W // blk, 2), dtype=np.int16)

    for by in range(0, H, blk):
        for bx in range(0, W, blk):
            best_cost = 1e18
            best_dy = 0
            best_dx = 0
            cur_blk = cur[by:by+blk, bx:bx+blk].astype(np.int16)

            # Reference top-left inside padded image
            ry = by + search_range
            rx = bx + search_range

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ref_blk = pr[ry + dy:ry + dy + blk, rx + dx:rx + dx + blk].astype(np.int16)
                    sad = np.abs(cur_blk - ref_blk).sum()
                    if sad < best_cost:
                        best_cost = sad
                        best_dy, best_dx = dy, dx

            # Write recon block and motion vector
            ref_blk_best = pr[ry + best_dy:ry + best_dy + blk, rx + best_dx:rx + best_dx + blk]
            recon[by:by+blk, bx:bx+blk] = ref_blk_best
            mv[by // blk, bx // blk] = (best_dy, best_dx)

    return recon, mv

def three_step_search_ME(ref, cur, blk=8, search_range=8):
    """
    Classic Three-Step Search (TSS):
      - Start at (0,0), step = max power of 2 <= search_range, then halve each step
      - Only 3 steps regardless of range
    Returns recon (uint8), mv (dy,dx) int16 per block
    """
    H, W = cur.shape
    pr = pad_for_search(ref, search_range)
    recon = np.zeros_like(cur, dtype=np.uint8)
    mv = np.zeros((H // blk, W // blk, 2), dtype=np.int16)

    # Initial step: largest power-of-two <= search_range, but >=1
    if search_range < 1:
        step0 = 1
    else:
        step0 = 1 << int(np.floor(np.log2(search_range)))

    for by in range(0, H, blk):
        for bx in range(0, W, blk):
            cur_blk = cur[by:by+blk, bx:bx+blk].astype(np.int16)
            ry = by + search_range
            rx = bx + search_range

            cy, cx = 0, 0  # current center (dy, dx)
            step = step0

            def cost_at(dy, dx):
                ref_blk = pr[ry + dy:ry + dy + blk, rx + dx:rx + dx + blk].astype(np.int16)
                return np.abs(cur_blk - ref_blk).sum()

            # Step 1..3
            for _ in range(3):
                best_cost = cost_at(cy, cx)
                best_dy, best_dx = cy, cx

                # Examine 8 neighbors (and center) with spacing "step"
                for dy in (-step, 0, step):
                    for dx in (-step, 0, step):
                        ny, nx = cy + dy, cx + dx
                        # Constrain to search window
                        if abs(ny) <= search_range and abs(nx) <= search_range:
                            c = cost_at(ny, nx)
                            if c < best_cost:
                                best_cost, best_dy, best_dx = c, ny, nx

                cy, cx = best_dy, best_dx
                step = max(1, step // 2)

            # Write output
            ref_blk_best = pr[ry + cy:ry + cy + blk, rx + cx:rx + cx + blk]
            recon[by:by+blk, bx:bx+blk] = ref_blk_best
            mv[by // blk, bx // blk] = (cy, cx)

    return recon, mv

# ------------------------------------------------------------
# Main comparison
# ------------------------------------------------------------
def run_all(ref_path="one_gray.png", cur_path="two_gray.png", blk=8):
    ref = ensure_gray_u8(cv2.imread(ref_path, cv2.IMREAD_UNCHANGED))
    cur = ensure_gray_u8(cv2.imread(cur_path, cv2.IMREAD_UNCHANGED))

    # Make sizes match and crop to block multiples
    H = min(ref.shape[0], cur.shape[0])
    W = min(ref.shape[1], cur.shape[1])
    ref = ref[:H, :W]
    cur = cur[:H, :W]
    ref = crop_to_block_multiple(ref, blk)
    cur = crop_to_block_multiple(cur, blk)

    print(f"Image size used: {cur.shape[1]}x{cur.shape[0]} (cropped to 8x8 blocks)")

    ranges = [8, 16, 32]

    # Full search comparisons
    for R in ranges:
        t0 = time.perf_counter()
        recon, mv = full_search_ME(ref, cur, blk=blk, search_range=R)
        dt = time.perf_counter() - t0
        p = psnr(cur, recon)
        print(f"[Full Search] range=±{R:>2} | PSNR={p:6.2f} dB | time={dt:7.3f} s")
        residual = cur.astype(np.int16) - recon.astype(np.int16)
        save_outputs(f"full_r{R}", recon, residual)

    # Three-step search (TSS) comparisons with same ranges
    for R in ranges:
        t0 = time.perf_counter()
        recon, mv = three_step_search_ME(ref, cur, blk=blk, search_range=R)
        dt = time.perf_counter() - t0
        p = psnr(cur, recon)
        print(f"[TSS]         range=±{R:>2} | PSNR={p:6.2f} dB | time={dt:7.3f} s")
        residual = cur.astype(np.int16) - recon.astype(np.int16)
        save_outputs(f"tss_r{R}", recon, residual)

if __name__ == "__main__":
    run_all("one_gray.png", "two_gray.png", blk=8)
