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
# DCT / IDCT (NumPy, 2D Accelerated)
# ==========================

def dct_matrix(N):
    """建立 DCT 轉換矩陣 (N×N)"""
    u = np.arange(N).reshape(-1, 1)
    x = np.arange(N).reshape(1, -1)
    alpha = np.sqrt(2 / N) * np.ones((N, 1))
    alpha[0, 0] = np.sqrt(1 / N)
    return alpha * np.cos(((2 * x + 1) * u * np.pi) / (2 * N))

def dct2_numpy(img):
    """2D-DCT = C * f * C.T"""
    N, M = img.shape
    C_N = dct_matrix(N)
    C_M = dct_matrix(M)
    return C_N @ img @ C_M.T

def idct2_numpy(dct_coeff):
    """2D-IDCT = C.T * F * C"""
    N, M = dct_coeff.shape
    C_N = dct_matrix(N)
    C_M = dct_matrix(M)
    return C_N.T @ dct_coeff @ C_M

# ==========================
# DCT / IDCT (Numpy, 1D Accelerated)
# ==========================

def dct1_numpy(signal):
    """1D DCT (Type-II) using matrix formulation"""
    N = signal.shape[0]
    C = dct_matrix(N)
    return C @ signal

def idct1_numpy(coeff):
    """1D IDCT (Inverse DCT) using matrix formulation"""
    N = coeff.shape[0]
    C = dct_matrix(N)
    return C.T @ coeff

def dct2_by_two_1d(img):
    """2D-DCT using two 1D-DCTs"""
    # Apply 1D-DCT on rows
    temp = np.apply_along_axis(dct1_numpy, axis=1, arr=img)
    # Apply 1D-DCT on columns
    return np.apply_along_axis(dct1_numpy, axis=0, arr=temp)

def idct2_by_two_1d(dct_coeff):
    """2D-IDCT using two 1D-IDCTs"""
    # Apply 1D-IDCT on columns
    temp = np.apply_along_axis(idct1_numpy, axis=0, arr=dct_coeff)
    # Apply 1D-IDCT on rows
    return np.apply_along_axis(idct1_numpy, axis=1, arr=temp)

# ==========================
# Main Function
# ==========================

def main():
    img = cv2.imread("./lena.png")
    print("Image shape:", img.shape)
    
    # Convert to grayscale
    print("Converting to grayscale...")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imsave("./gray_image.png", gray_img, cmap='gray')
    print("Grayscale image is saved to ./gray_image.png")
    
    root_outdir = "./figures"
    os.makedirs(root_outdir, exist_ok=True)
    print(f"The DCT figures will be saved to: {root_outdir}")
    
    # ==========================
    # Fast DCT / IDCT Implementation (NumPy Accelerated)
    # ==========================
    print("\n ===== Fast DCT / IDCT Implementation (NumPy Accelerated) =====\n")
    output_dir = root_outdir + '/dct_fast_numpy'
    os.makedirs(output_dir, exist_ok=True)
    
    # DCT
    print("Applying Fast(NumPy) 2D DCT...")
    start_time = time.time()
    dct_img_numpy = dct2_numpy(gray_img)
    end_time = time.time()
    print(f"[Fast(NumPy) DCT runtime]: {end_time - start_time:.2f}s")
    
    # === visualize in log domain ===
    dct_log_numpy = np.log1p(np.abs(dct_img_numpy))
    dct_log_numpy = dct_log_numpy / np.max(dct_log_numpy)
    plt.imsave(f"{output_dir}/dct_log_image.png", dct_log_numpy, cmap='gray')
    print(f"Fast(NumPy) DCT log-domain image saved to {output_dir}/dct_log_image.png\n")

    # IDCT
    print("Applying Fast(NumPy) IDCT...")
    start_time = time.time()
    idct_img_numpy = idct2_numpy(dct_img_numpy)
    end_time = time.time()
    print(f"[Fast(NumPy) IDCT runtime]: {end_time - start_time:.2f}s")

    # Save reconstructed image
    idct_img_numpy_clipped = np.clip(idct_img_numpy, 0, 255)
    plt.imsave(f"{output_dir}/idct_image.png", idct_img_numpy_clipped, cmap='gray')
    print(f"Fast(NumPy) IDCT image is saved to {output_dir}/idct_image.png\n")

    # === PSNR ===
    psnr_value_numpy = compute_psnr(gray_img, idct_img_numpy_clipped)
    print(f"PSNR (Fast(NumPy) implementation): {psnr_value_numpy:.2f} dB")

    # ==========================
    # Fast DCT / IDCT Implementation (1D Accelerated)
    # ==========================
    print("\n ===== Fast DCT / IDCT Implementation (1D Accelerated) =====\n")
    output_dir = root_outdir + '/dct_fast_1d'
    os.makedirs(output_dir, exist_ok=True)

    # DCT
    print("Applying Fast(1D) 2D DCT...")
    start_time = time.time()
    dct_img_fast_1d = dct2_by_two_1d(gray_img)
    end_time = time.time()
    print(f"[Fast(1D) DCT runtime]: {end_time - start_time:.2f}s")

    # === visualize in log domain ===
    dct_log_fast_1d = np.log1p(np.abs(dct_img_fast_1d))
    dct_log_fast_1d = dct_log_fast_1d / np.max(dct_log_fast_1d)
    plt.imsave(f"{output_dir}/dct_log_image.png", dct_log_fast_1d, cmap='gray')
    print(f"Fast(1D) DCT log-domain image saved to {output_dir}/dct_log_image.png\n")

    # IDCT
    print("Applying Fast(1D) IDCT...")
    start_time = time.time()
    idct_img_fast_1d = idct2_by_two_1d(dct_img_fast_1d)
    end_time = time.time()
    print(f"[Fast(1D) IDCT runtime]: {end_time - start_time:.2f}s")

    # Save reconstructed image
    idct_img_fast_1d_clipped = np.clip(idct_img_fast_1d, 0, 255)
    plt.imsave(f"{output_dir}/idct_image.png", idct_img_fast_1d_clipped, cmap='gray')
    print(f"Fast(1D) IDCT image is saved to {output_dir}/idct_image.png\n")

    # === PSNR ===
    psnr_value_fast_1d = compute_psnr(gray_img, idct_img_fast_1d_clipped)
    print(f"PSNR (Fast(1D) implementation): {psnr_value_fast_1d:.2f} dB")

if __name__ == "__main__":
    main()