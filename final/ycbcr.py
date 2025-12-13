# ycbcr.py
import numpy as np

def ycbcr_to_rgb_formula(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255)

def ycbcr_to_rgb_table(Y, Cb, Cr):
    # lookup table approximation
    Cr_lut = 1.402 * (Cr - 128)
    Cb_lut = 1.772 * (Cb - 128)
    G_lut = -0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)

    R = Y + Cr_lut
    G = Y + G_lut
    B = Y + Cb_lut
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255)
