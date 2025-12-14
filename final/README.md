# Final: Jpeg Decoder Implementation & Performance Analysis
**Student ID:** [314554025]  
**Name:** [鄭睿宏 / 郭彥頡 / 孫承瑞 / 戚維凌]  
**Project Link:** [https://github.com/RyanCheng98153/video-compression/tree/main/final](https://github.com/RyanCheng98153/video-compression/tree/main/final)

## Introduction
This project implements a Baseline JPEG Decoder with configurable components for performance and quality analysis.
It allows you to perform ablation studies on:
- Dequantization precision
- IDCT implementation
- YCbCr → RGB conversion method

and evaluate their impact using PSNR, SSIM, and runtime statistics.

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

Just use the defined `run.sh`: 

```bash
bash ./run.sh
```

Use main.py as the entry point:

```bash
python main.py \
  --png <ground_truth_png> \
  --jpg <input_jpeg> \
  --ycbcr <formula|table> \
  --idct <2d|two1d|block> \
  --dequant <float|int> \
  --out_img_dir <output_dir>
```

### Usage Arguments

Input Files: 
  - `--png`: Ground-truth PNG image.
Used for quality evaluation (PSNR / SSIM).

  - `--jpg`: Input JPEG image to be decoded (Baseline JPEG with Huffman coding).

Decoder Options (Ablation Settings): 
  - `--dequant`: Controls the dequantization precision:
    
    - `float`: Floating-point dequantization (higher accuracy)
    - `int`: Integer dequantization (faster, lower precision)
  
  - `--idct`: Selects the IDCT implementation:
    
    - `2d`: Direct 2D IDCT implementation
    - `two1d`: Two-pass 1D IDCT (row-wise + column-wise)
  
  - `--ycbcr`: Controls the YCbCr → RGB conversion method:
    
    - `formula`: Standard formula-based conversion
    - `table`: Lookup-table-based conversion (faster)

Output
  - `--out_img_dir`: Directory where decoded images will be saved.
    - example format in ablation: `<ycbcr>_<idct>_<dequant>.png`
    - Example: 
      - `formula_2d_float.png`
      - `table_block_int.png`

> [!NOTE]
> All the results will be stored in `results/<image>/` dir.
> All the decoded images will be stored in `results/<image>/result_images/` dir

Output Directory Structure
results/`<image>`/
├── result_images/
│   ├── formula_2d_float.png
│   ├── table_two1d_int.png
│   └── ...

## License

This project is for educational purposes.