# quantization.py
import numpy as np

def dequantize_blocks(blocks_q: np.ndarray,
                      qtable: np.ndarray,
                      mode: str = "float") -> np.ndarray:
    """
    Dequantization ablation

    blocks_q : (H_blocks, W_blocks, 8, 8)
               quantized DCT coefficients (float32, integer-valued)
    qtable   : (8, 8) quantization table parsed from JPEG bitstream
    mode     : 'float' | 'int'

    Returns
    -------
    blocks : float32 dequantized coefficients
    """

    if mode == "float":
        # Baseline: float32 multiply
        return blocks_q * qtable[None, None, :, :]

    elif mode == "int":
        # Integer-style dequantization (hardware-like)
        # int16 * int16 -> int32 -> float32
        b = blocks_q.astype(np.int16)
        q = qtable.astype(np.int16)
        out = b * q[None, None, :, :]
        return out.astype(np.float32)

    else:
        raise ValueError(f"Unknown dequant mode: {mode}")
