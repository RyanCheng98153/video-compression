# ycbcr.py
import numpy as np

# Build LUT once (0..255)
# Use int16 to avoid overflow, then clip at the end
_cr = (np.arange(256) - 128).astype(np.int16)
_cb = (np.arange(256) - 128).astype(np.int16)

# JPEG standard-ish coefficients
LUT_CR_R = np.round(1.402 * _cr).astype(np.int16)
LUT_CB_B = np.round(1.772 * _cb).astype(np.int16)
LUT_CB_G = np.round(0.344136 * _cb).astype(np.int16)
LUT_CR_G = np.round(0.714136 * _cr).astype(np.int16)

def ycbcr_to_rgb_formula(Y, Cb, Cr):
    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32)
    Cr = Cr.astype(np.float32)

    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255)

def ycbcr_to_rgb_table(Y, Cb, Cr):
    """
    Table lookup version:
    R = Y + LUT_CR_R[Cr]
    G = Y - LUT_CB_G[Cb] - LUT_CR_G[Cr]
    B = Y + LUT_CB_B[Cb]
    """
    Y_i = Y.astype(np.int16)
    Cb_u = Cb.astype(np.uint8)
    Cr_u = Cr.astype(np.uint8)

    R = Y_i + LUT_CR_R[Cr_u]
    G = Y_i - LUT_CB_G[Cb_u] - LUT_CR_G[Cr_u]
    B = Y_i + LUT_CB_B[Cb_u]

    rgb = np.stack([R, G, B], axis=-1).astype(np.int16)
    return np.clip(rgb, 0, 255).astype(np.float32)
