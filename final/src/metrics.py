# metrics.py
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

def compute_metrics(gt, pred):
    psnr = peak_signal_noise_ratio(gt, pred, data_range=255)
    ssim = structural_similarity(gt, pred, channel_axis=-1, data_range=255)
    return psnr, ssim
