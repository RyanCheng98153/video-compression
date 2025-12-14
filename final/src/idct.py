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
    return idct(
        idct(block, axis=0, norm="ortho"),
        axis=1,
        norm="ortho"
    )


# ============================================================
# 2. Two 1D IDCT (row-wise then column-wise)
# ============================================================
def idct_two_1d(block: np.ndarray) -> np.ndarray:
    """
    Perform IDCT by applying 1D IDCT twice (rows then columns).

    Input:
        block: (8, 8) dequantized DCT coefficients (float32)

    Output:
        (8, 8) spatial domain block (float32)
    """
    tmp = np.zeros_like(block, dtype=np.float32)
    out = np.zeros_like(block, dtype=np.float32)

    # Row-wise IDCT
    for i in range(8):
        tmp[i, :] = idct(block[i, :], norm="ortho")

    # Column-wise IDCT
    for j in range(8):
        out[:, j] = idct(tmp[:, j], norm="ortho")

    return out


# ============================================================
# 3. Block-based IDCT (full image)
# ============================================================
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
            ] = idct2d(block)

    return image
