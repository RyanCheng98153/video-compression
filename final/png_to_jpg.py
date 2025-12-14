# png_to_jpg.py
from PIL import Image
import argparse
import os

def png_to_jpg(png_path, jpg_path, quality=95):
    img = Image.open(png_path).convert("RGB")
    # img.save(jpg_path, "JPEG", quality=quality, subsampling=0)
    img.save("lena.jpg", "JPEG", quality=95, subsampling=0, progressive=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", required=True)
    parser.add_argument("--jpg", required=True)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    png_to_jpg(args.png, args.jpg, args.quality)
