# Image Compression HW1

This project provides tools for image color transformation, supporting the following features:

- Convert images between multiple color spaces:
    - **RGB**
    - **YUV**
    - **YCbCr**
- Generate grayscale images from each color space.
- Optional acceleration using NumPy for improved performance.

The implementation is intended for educational use and demonstrates fundamental concepts in image processing.

## Installation (uv or pip)

### Using [uv](https://github.com/astral-sh/uv) (recommended)
```bash
uv init
uv sync
```
Activate the virtual environment:
- On Windows:
    ```bash
    ./.venv/Scripts/Activate.ps1
    ```
- On macOS/Linux:
    ```bash
    source .venv/bin/activate
    ```

### Using pip (if does not have uv)
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

For NumPy-accelerated version:
```bash
python numpy-main.py
```

> [!NOTE]
> Image files will be generated in the `output` directory.

## File Structure
.
├── main.py
├── numpy-main.py
├── lena.png
├── output/
│   ├── rgb_R.png
│   ├── rgb_G.png
│   ├── rgb_B.png
│   ├── yuv_Y.png
│   ├── yuv_U.png
│   ├── yuv_V.png
│   ├── ycbcr_Y.png
│   ├── ycbcr_Cb.png
│   ├── ycbcr_Cr.png
├── report.md
├── README.md
├── requirements.txt


## Requirements

All dependencies are listed in `requirements.txt`.

- numpy
- opencv-python

## License

This project is for educational purposes.
