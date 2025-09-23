import cv2
import numpy as np
import os

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR (OpenCV default) to RGB.
    Just flip the last axis.
    """
    return img[..., ::-1]

def convert_YUV(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB to YUV (ITU-R BT.601 standard).
    Formula:
        Y = 0.299R + 0.587G + 0.114B
        U = -0.169R - 0.331G + 0.5B + 128
        V = 0.5R - 0.419G - 0.081B + 128
    """
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.169 * R - 0.331 * G + 0.5 * B + 128
    V = 0.5 * R - 0.419 * G - 0.081 * B + 128

    return np.clip(np.dstack((Y, U, V)), 0, 255).astype(np.uint8)

def convert_YCbCr(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB to YCbCr (JPEG standard).
    Formula:
        Y  = 0.299R + 0.587G + 0.114B
        Cb = -0.168736R - 0.331264G + 0.5B + 128
        Cr = 0.5R - 0.418688G - 0.081312B + 128
    """
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

    return np.clip(np.dstack((Y, Cb, Cr)), 0, 255).astype(np.uint8)

def save_channel_gray(img: np.ndarray, prefix: str, channel_names: list[str]):
    """
    Save each channel as a grayscale image.
    Each channel is expanded into (H, W, 3) so OpenCV can save it.
    """
    for i, name in enumerate(channel_names):
        channel = img[..., i]
        gray_img = np.dstack((channel, channel, channel))
        cv2.imwrite(f"./output/{prefix}_{name}.png", gray_img)

def main():
    # 讀圖 (OpenCV 預設是 BGR)
    img_bgr = cv2.imread("./lena.png")
    img_rgb = bgr_to_rgb(img_bgr)

    # 建立輸出資料夾
    os.makedirs("./output", exist_ok=True)

    # RGB 通道
    save_channel_gray(img_rgb, "rgb", ["R", "G", "B"])

    # YUV 通道
    img_yuv = convert_YUV(img_rgb)
    save_channel_gray(img_yuv, "yuv", ["Y", "U", "V"])

    # YCbCr 通道
    img_ycbcr = convert_YCbCr(img_rgb)
    save_channel_gray(img_ycbcr, "ycbcr", ["Y", "Cb", "Cr"])

    print("All images have been saved in ./output/")

if __name__ == "__main__":
    main()
