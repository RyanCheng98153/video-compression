# idct.py
import numpy as np
from scipy.fftpack import idct

def idct2d(block):
    return idct(idct(block, axis=0, norm="ortho"), axis=1, norm="ortho")

def idct_two_1d(block):
    tmp = np.zeros_like(block)
    for i in range(8):
        tmp[i, :] = idct(block[i, :], norm="ortho")
    out = np.zeros_like(tmp)
    for j in range(8):
        out[:, j] = idct(tmp[:, j], norm="ortho")
    return out

def idct_block_based(blocks):
    """
    blocks: (H/8, W/8, 8, 8)
    """
    h, w, _, _ = blocks.shape
    out = np.zeros((h*8, w*8))
    for i in range(h):
        for j in range(w):
            out[i*8:(i+1)*8, j*8:(j+1)*8] = idct2d(blocks[i, j])
    return out
