# HW2 – DCT Report
**Name:** 鄭睿宏  
**Student ID:** 314554025 
**Assignment:** 2D-DCT and Two 1D-DCT Implementation on “lena.png”

## 📘 1. Introduction

In this homework, I implemented both the `2D Discrete Cosine Transform (2D-DCT)` and the `two 1D-DCT (accelerated)` approach, and use `Peak Signal-to-Noise Ratio (PSNR)` as the evaluation metric.

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

Run the main script (For Experiment Observation):
```bash
python main.py 
```
- Note: Use `python main.py --uv_size 64` to discard high frequency detail in DCT.
  - `--uv_size`: can choose from (4, 8, 16, 32, ..., 512(max))

For NumPy-accelerated version (Fast and Recommeded):
```bash
python numpy_main.py
```

> [!NOTE]
> Image files will be generated in the `./figures` directory.

## License

This project is for educational purposes.
