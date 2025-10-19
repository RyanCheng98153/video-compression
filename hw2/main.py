import os
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === PSNR computation ===
def compute_psnr(original, reconstructed):
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed images must have the same dimensions.")
    
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# ==========================
# DCT / IDCT (Basic Implementation)
# ==========================

# calculate value in DCT matrix F-formula
def cal_dct2_F(gray_img, u, v):
    # calculate the discrete cosine transform value at position (u, v)
    N, M = gray_img.shape
    sum_value = 0.0
    
    for x in range(N):
        for y in range(M):
            sum_value += (
                gray_img[x, y] 
                * math.cos((2 * x + 1) * u * math.pi / (2 * N)) 
                * math.cos((2 * y + 1) * v * math.pi / (2 * M))
            )
    Cu = 1 / math.sqrt(2) if u == 0 else 1.0
    Cv = 1 / math.sqrt(2) if v == 0 else 1.0
    
    coeff = (2 / math.sqrt(N * M)) * Cu * Cv
    return coeff * sum_value

def dct2(gray_img, u_size=None, v_size=None):
    N, M = gray_img.shape
    dct_img = np.zeros((N, M), dtype=np.float32)

    # A compression option: only compute up to u_size and v_size
    u_max = u_size if u_size is not None and u_size < N else N
    v_max = v_size if v_size is not None and v_size < M else M

    for u in tqdm(range(u_max), desc="Calculating DCT"):
        for v in tqdm(range(v_max), desc=f"Calculating DCT row {u+1}/{u_max}"):
            dct_img[u, v] = cal_dct2_F(gray_img, u, v)
    return dct_img

# === IDCT implementation ===

def cal_idct2_f(dct_img, x, y, u_max=None, v_max=None):
    # Mostly the M, N are equal
    N, M = dct_img.shape
    
    # No need to calculate number more than u_size and v_size, since they are zeroed out
    u_max = u_max if u_max is not None and u_max <= N else N
    v_max = v_max if v_max is not None and v_max <= M else M

    sum_value = 0.0
    for u in range(u_max):
        for v in range(v_max):
            Cu = 1 / math.sqrt(2) if u == 0 else 1.0
            Cv = 1 / math.sqrt(2) if v == 0 else 1.0
            coeff = (2 / math.sqrt(N * M)) * Cu * Cv
            sum_value += (
                coeff 
                * dct_img[u, v] 
                * math.cos((2 * x + 1) * u * math.pi / (2 * N)) 
                * math.cos((2 * y + 1) * v * math.pi / (2 * M))
            )
    return sum_value

def idct2(dct_img, u_size=None, v_size=None):
    N, M = dct_img.shape
    
    idct_img = np.zeros((N, M), dtype=np.float32)
    for x in tqdm(range(N), desc="Calculating IDCT (rows)"):
        for y in range(M):
            idct_img[x, y] = cal_idct2_f(dct_img, x, y, u_max=u_size, v_max=v_size)
    return idct_img

# ==========================
# DCT / IDCT (1D Accelerated)
# ==========================

# === 1D DCT along a single axis ===
def dct_1d(vector, k_max=None):
    N = len(vector)
    result = np.zeros(N, dtype=np.float32)
    
    # No need to calculate more than k_max
    k_max = k_max if k_max is not None and k_max <= N else N
    
    for k in range(k_max):
        coeff = 1 / math.sqrt(2) if k == 0 else 1.0
        sum_val = 0.0
        for n in range(N):
            sum_val += vector[n] * math.cos((2*n + 1) * k * math.pi / (2 * N))
        result[k] = math.sqrt(2/N) * coeff * sum_val
    return result

# === Two 1D-DCT ===
def dct2_fast(img, u_size=None, v_size=None):
    N, M = img.shape
    temp = np.zeros((N, M), dtype=np.float32)
    dct_img = np.zeros((N, M), dtype=np.float32)

    # Step 1: DCT along rows
    for i in tqdm(range(N), desc="Calculating DCT (rows)"):
        temp[i, :] = dct_1d(img[i, :], v_size)
    
    # Step 2: DCT along columns
    for j in tqdm(range(M), desc="Calculating DCT (columns)"):
        dct_img[:, j] = dct_1d(temp[:, j], u_size)
    
    return dct_img

# === Two 1D-IDCT ===
def idct_1d(vector, k_max=None):
    N = len(vector)
    result = np.zeros(N, dtype=np.float32)
    
    # No need to calculate more than k_max
    k_max = k_max if k_max is not None and k_max <= N else N
    
    for n in range(N):
        sum_val = 0.0
        for k in range(k_max):
            coeff = 1 / math.sqrt(2) if k == 0 else 1.0
            sum_val += coeff * vector[k] * math.cos((2*n + 1) * k * math.pi / (2 * N))
        result[n] = math.sqrt(2/N) * sum_val
    return result

def idct2_fast(dct_img, u_size=None, v_size=None):
    N, M = dct_img.shape
    # Step 1: IDCT along columns
    temp = np.zeros((N, M), dtype=np.float32)
    for j in tqdm(range(M), desc="Calculating IDCT (columns)"):
        temp[:, j] = idct_1d(dct_img[:, j], k_max=u_size)
    
    # Step 2: IDCT along rows
    idct_img = np.zeros((N, M), dtype=np.float32)
    for i in tqdm(range(N), desc="Calculating IDCT (rows)"):
        idct_img[i, :] = idct_1d(temp[i, :], k_max=v_size)
    
    return idct_img

# ==========================
# Main Function
# ==========================

def write_log(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def main(
    uv_size: int =None
):
    img = cv2.imread("./lena.png")
    print("Image shape:", img.shape)
    log_file = "./dct_result.log"

    # Convert to grayscale
    print("Converting to grayscale...")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imsave("./gray_image.png", gray_img, cmap='gray')
    print("Grayscale image is saved to ./gray_image.png")
    
    root_outdir = "./figures"
    os.makedirs(root_outdir, exist_ok=True)
    print(f"The DCT figures will be saved to: {root_outdir}")

    # ==========================
    # Basic DCT / IDCT Implementation
    # ==========================
    print("\n ===== Basic DCT / IDCT Implementation =====\n")

    output_dir = root_outdir + '/dct_basic'
    os.makedirs(output_dir, exist_ok=True)

    u_size, v_size = uv_size, uv_size
    uv_size = img.shape[0] if uv_size is None else uv_size
    
    # DCT
    print(f"Applying Basic 2D DCT... with u_size={u_size}, v_size={v_size}")
    start_time = time.time()
    dct_img = dct2(gray_img, u_size=u_size, v_size=v_size)
    print(f"\n[Basic DCT runtime, uv_size {u_size}]: {time.time() - start_time:.2f} s")
    write_log(log_file, f"[Basic DCT runtime, uv_size {u_size}]: {time.time() - start_time:.2f} s")

    # === visualize in log domain ===
    dct_log = np.log1p(np.abs(dct_img))
    dct_log = dct_log / np.max(dct_log)
    plt.imsave(f"{output_dir}/dct_log_image_uv_{uv_size}.png", dct_log, cmap='gray')
    print(f"DCT log-domain image saved to {output_dir}/dct_log_image_uv_{uv_size}.png\n")

    # IDCT
    print("Applying Basic IDCT...")
    start_time = time.time()
    idct_img = idct2(dct_img, u_size=u_size, v_size=v_size)
    print(f"\n[Basic IDCT runtime, uv_size {u_size}]: {time.time() - start_time:.2f} s")
    write_log(log_file, f"[Basic IDCT runtime, uv_size {u_size}]: {time.time() - start_time:.2f} s")

    # Save reconstructed image
    idct_img_clipped = np.clip(idct_img, 0, 255)
    plt.imsave(f"{output_dir}/idct_image_uv_{uv_size}.png", idct_img_clipped, cmap='gray')
    print(f"IDCT image is saved to {output_dir}/idct_image_uv_{uv_size}.png\n")

    # === PSNR ===
    psnr_value = compute_psnr(gray_img, idct_img_clipped)
    print(f"[Basic PSNR, uv_size {u_size}]: {psnr_value:.2f} dB")
    write_log(log_file, f"[Basic PSNR, uv_size {u_size}]: {psnr_value:.2f} dB\n")
    
    # ==========================
    # Fast DCT / IDCT Implementation (1D Accelerated)
    # ==========================
    print("\n ===== Fast DCT / IDCT Implementation (1D Accelerated) =====\n")
    output_dir = root_outdir + '/dct_fast_1d'
    os.makedirs(output_dir, exist_ok=True)

    # DCT
    print(f"Applying Fast(1D) 2D DCT... with u_size={u_size}, v_size={v_size}")
    start_time = time.time()
    dct_img_fast = dct2_fast(gray_img, u_size=u_size, v_size=v_size)
    print(f"\n[Fast(1D) DCT runtime, uv_size {u_size}]: {time.time() - start_time:.2f} s")
    write_log(log_file, f"[Fast(1D) DCT runtime, uv_size {u_size}]: {time.time() - start_time:.2f} s")

    # === visualize in log domain ===
    dct_log_fast = np.log1p(np.abs(dct_img_fast))
    dct_log_fast = dct_log_fast / np.max(dct_log_fast)
    plt.imsave(f"{output_dir}/dct_log_image_uv_{uv_size}.png", dct_log_fast, cmap='gray')
    print(f"Fast(1D) DCT log-domain image saved to {output_dir}/dct_log_image_uv_{uv_size}.png\n")

    # IDCT
    print("Applying Fast(1D) IDCT...")
    start_time = time.time()
    idct_img_fast = idct2_fast(dct_img_fast, u_size=u_size, v_size=v_size)
    print(f"\n[Fast(1D) IDCT runtime, uv_size {u_size}]: {time.time() - start_time:.2f} s")
    write_log(log_file, f"[Fast(1D) IDCT runtime, uv_size {u_size}]: {time.time() - start_time:.2f} s")

    # Save reconstructed image
    idct_img_fast_clipped = np.clip(idct_img_fast, 0, 255)
    plt.imsave(f"{output_dir}/idct_image_uv_{uv_size}.png", idct_img_fast_clipped, cmap='gray')
    print(f"Fast(1D) IDCT image is saved to {output_dir}/idct_image_uv_{uv_size}.png\n")

    # === PSNR ===
    psnr_value_fast = compute_psnr(gray_img, idct_img_fast_clipped)
    print(f"[Fast(1D) PSNR, uv_size {u_size}]: {psnr_value_fast:.2f} dB")
    write_log(log_file, f"[Fast(1D) PSNR, uv_size {u_size}]: {psnr_value_fast:.2f} dB\n")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCT and IDCT Implementations")
    parser.add_argument('--uv_size', type=int, default=None, help='Size of u and v for DCT computation')
    args = parser.parse_args()

    main(uv_size=args.uv_size)