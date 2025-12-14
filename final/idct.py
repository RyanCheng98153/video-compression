import numpy as np
from scipy.fftpack import idct

def idct2d(block: np.ndarray) -> np.ndarray:
    return idct(idct(block, axis=0, norm="ortho"), axis=1, norm="ortho")

def idct_two_1d(block: np.ndarray) -> np.ndarray:
    tmp = np.zeros_like(block)
    for i in range(8):
        tmp[i, :] = idct(block[i, :], norm="ortho")
    out = np.zeros_like(tmp)
    for j in range(8):
        out[:, j] = idct(tmp[:, j], norm="ortho")
    return out

def idct_blocks(blocks: np.ndarray, method: str) -> np.ndarray:
    """
    blocks: (bh, bw, 8, 8)
    return: same shape, spatial-domain blocks
    """
    bh, bw, _, _ = blocks.shape
    out = np.empty_like(blocks, dtype=np.float32)

    if method == "2d":
        for i in range(bh):
            for j in range(bw):
                out[i, j] = idct2d(blocks[i, j])

    elif method == "two1d":
        for i in range(bh):
            for j in range(bw):
                out[i, j] = idct_two_1d(blocks[i, j])

    elif method == "blocked":
        # baseline "blocked pipeline": still per-block, but separated as a pipeline method
        # (you can later optimize this path for speed and compare)
        for i in range(bh):
            for j in range(bw):
                out[i, j] = idct2d(blocks[i, j])

    else:
        raise ValueError("Unknown IDCT method. Use: 2d / two1d / blocked")

    return out
