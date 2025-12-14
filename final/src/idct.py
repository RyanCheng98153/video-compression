# idct.py
import numpy as np
from scipy.fftpack import idct

# ============================================================
# 1. Standard 2D IDCT (per 8x8 block)
# ============================================================
def idct2d(block: np.ndarray) -> np.ndarray:
    """
    Perform 2D IDCT on a single 8x8 block.

    Input:
        block: (8, 8) dequantized DCT coefficients (float32)

    Output:
        (8, 8) spatial domain block (float32)
    """
    N = block.shape[0]
    img_float = block.astype(np.float32)

    C = np.ones(N, dtype=np.float32)
    C[0] = 1 / np.sqrt(2)
    C *= np.sqrt(2 / N)

    XY = np.arange(N)[None, :]      # (1, N), row
    UV = np.arange(N)[:, None]      # (N, 1), col
    cos_table = np.cos((2 * XY + 1) * UV * np.pi / (2 * N))  # (N, N)
    cos_table *= C[:, None]         # colume-wise

    result = np.zeros((N, N), dtype=np.float32)
    for x in range(N):
        cos_u = cos_table[:, x]         # fix x, u = [0, N-1]
        for y in range(N):
            cos_v = cos_table[:, y]     # fix y, v = [0, N-1]
            cos_uv = cos_u[:, None] * cos_v[None, :]    # (N, N)
            result[x, y] = np.sum(img_float * cos_uv)

    return result


# ============================================================
# 2. Two 1D IDCT (row-wise then column-wise)
# ============================================================

def basis1D(N):
    X = np.arange(N, dtype=np.float32)[None, :]         # (1, N)
    U = np.arange(N, dtype=np.float32)[:, None]         # (N, 1)

    C = np.ones((N, 1), dtype=np.float32)               # (N, 1)
    C[0, 0] = 1 / np.sqrt(2)

    basis = np.cos((2 * X + 1) * U * np.pi / (2 * N))   # (N, N)
    
    return basis * (np.sqrt(2 / N) * C)                 # (N, N)


def idct_two_1d(block: np.ndarray) -> np.ndarray:
    """
    Perform IDCT by applying 1D IDCT twice (rows then columns).

    Input:
        block: (8, 8) dequantized DCT coefficients (float32)

    Output:
        (8, 8) spatial domain block (float32)
    """
    N = block.shape[0]
    basis = basis1D(N)
    first = block @ basis     # row
    result = basis.T @ first    # col

    return result


# ============================================================
# 3. Block-based IDCT (full image)
# ============================================================

def idct_block(block):
    return idct(
        idct(block, axis=0, norm="ortho"),
        axis=1,
        norm="ortho"
    )

def idct_block_based(blocks: np.ndarray) -> np.ndarray:
    """
    Apply IDCT block-by-block and stitch into a full image.

    Input:
        blocks: (bh, bw, 8, 8)
                dequantized DCT coefficients

    Output:
        image: (bh*8, bw*8)
               reconstructed spatial image
    """
    bh, bw, _, _ = blocks.shape
    H = bh * 8
    W = bw * 8

    image = np.zeros((H, W), dtype=np.float32)

    for by in range(bh):
        for bx in range(bw):
            block = blocks[by, bx]
            image[
                by * 8 : (by + 1) * 8,
                bx * 8 : (bx + 1) * 8
            ] = idct_block(block)

    return image
