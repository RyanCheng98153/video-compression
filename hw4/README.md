# HW4: JPEG-like Compression with RLE and DCT
**Student ID:** [314554025]  
**Name:** [鄭睿宏]  
**Project Link:** [https://github.com/RyanCheng98153/video-compression/tree/main/hw4](https://github.com/RyanCheng98153/video-compression/tree/main/hw4)

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
python ./main.py --infile ./lena.png
```

> [!NOTE]
> Image files will be generated in the `./figures` directory.

## License

This project is for educational purposes.
