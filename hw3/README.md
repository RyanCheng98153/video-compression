# Homework 3 Report: Motion Estimation
**Student ID:** [314554025]  
**Name:** [鄭睿宏]  
**Project Link:** https://github.com/RyanCheng98153/video-compression/tree/main/hw3

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
python main.py --ref_img ./one_gray.png --cur_img ./two_gray.png
```

> [!NOTE]
> Image files will be generated in the `./figures` directory.

## License

This project is for educational purposes.
