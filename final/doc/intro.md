# JPEG Decoder Implementation and Ablation Study

## Overview

In this project, I implemented a **baseline JPEG decoder** from scratch in Python and conducted a systematic **ablation study** on several key components of the decoding pipeline.

The decoder operates **directly on the JPEG bitstream** and follows the **ITU-T T.81 / ISO JPEG Baseline (Sequential, Huffman-coded)** standard.  
All ablation studies are performed **without breaking JPEG correctness**, ensuring meaningful PSNR, SSIM, and runtime comparisons.

---

## Overall JPEG Decoding Pipeline

The complete decoding pipeline implemented in this project is shown below:

```
JPEG Bitstream
→ Marker Parsing (DQT, DHT, SOF0, SOS)
→ Entropy Decoding (Huffman)
→ ZigZag Reordering
→ Dequantization
→ Inverse DCT (IDCT)
→ Level Shift (+128)
→ YCbCr → RGB Conversion
→ Image Reconstruction
```


Each stage is explicitly implemented and exposed for ablation where possible.

---

## 1. JPEG Marker Parsing (Bitstream-Level Decoder)

A custom JPEG marker parser is implemented to directly read and interpret the JPEG bitstream.

### Supported Markers
- **SOF0 (Start of Frame)**
  - Image width and height
  - Component IDs and sampling factors
  - Quantization table assignment
- **DQT (Define Quantization Table)**
  - Parse 8×8 quantization tables
  - Reordered using JPEG ZigZag order
- **DHT (Define Huffman Table)**
  - Parse DC and AC Huffman tables
  - Construct canonical Huffman decoding structures
- **SOS (Start of Scan)**
  - Component scan order
  - Huffman table selection
- **Restart Markers (RST0–RST7)**
  - Properly handled during entropy decoding

This enables decoding of **real JPEG images**, not pre-extracted coefficient data.

---

## 2. Entropy Decoding (Huffman Decoding)

The entropy decoding stage strictly follows the JPEG baseline specification.

### Implemented Features
- **DC coefficient decoding**
  - Huffman decode category
  - Receive-and-extend to signed integer
  - Differential decoding using previous DC value
- **AC coefficient decoding**
  - Run-length decoding
  - End-of-Block (EOB) handling
  - Zero Run-Length (ZRL) handling

Each Minimum Coded Unit (MCU) is reconstructed as an 8×8 block of DCT coefficients.

---

## 3. ZigZag Reordering

ZigZag reordering is implemented according to the JPEG standard to map the 1D AC coefficient stream back to a 2D 8×8 frequency block.

> **Note:**  
> ZigZag reordering is **mandatory** for correct JPEG decoding.  
> Early experiments showed that disabling ZigZag leads to severe artifacts and decoding failure.

Therefore, ZigZag is **fixed ON** and not treated as an ablation dimension.

---

## 4. Dequantization (Ablation Dimension)

After entropy decoding, quantized DCT coefficients are dequantized using the **quantization tables parsed directly from the JPEG bitstream**.

Two dequantization implementations are compared:

### Dequantization Methods
1. **Float Dequantization**
   - `float32 × float32`
   - Straightforward software implementation
2. **Integer-style Dequantization**
   - `int16 × int16 → int32 → float32`
   - Mimics hardware-friendly or fixed-point decoders

Both methods use identical quantization tables; only arithmetic precision differs.

---

## 5. Inverse Discrete Cosine Transform (IDCT) (Ablation Dimension)

Three IDCT implementations are provided:

1. **Direct 2D IDCT**
   - Apply a single 2D IDCT per 8×8 block
2. **Two-pass 1D IDCT**
   - Row-wise 1D IDCT followed by column-wise 1D IDCT
   - Mathematically equivalent to 2D IDCT
3. **Block-based IDCT**
   - Perform IDCT block-by-block and explicitly stitch the full image

Although mathematically equivalent, these methods exhibit different computational characteristics.

---

## 6. Level Shift (Critical JPEG Step)

After IDCT, all values are **level-shifted by +128**:

```
Y, Cb, Cr := IDCT_output + 128
```


This step converts the zero-centered IDCT output into valid pixel values within `[0, 255]`.

Omitting this step leads to severe color distortion (e.g., green-tinted output), which was observed and corrected during development.

---

## 7. YCbCr to RGB Conversion (Ablation Dimension)

Two YCbCr → RGB conversion methods are implemented:

1. **Formula-based Conversion**
   - Direct floating-point computation using JPEG standard coefficients
2. **Lookup Table (LUT) Conversion**
   - Precomputed integer lookup tables
   - More efficient and hardware-friendly

Both methods produce visually equivalent results with minimal numerical differences.

---

## 8. Evaluation and Measurement

Each decoder configuration is evaluated using:

- **Runtime Performance**
  - Mean, standard deviation, and total runtime over 10 runs
- **Image Quality**
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
- **Visual Inspection**
  - Reconstructed RGB images saved for comparison

---

## Ablation Study Summary

The final ablation study explores the following dimensions:

| Component | Variants |
|---|---|
| YCbCr Conversion | Formula / LUT |
| IDCT | 2D / Two-pass 1D / Block-based |
| Dequantization | Float / Integer |

This results in **12 valid JPEG decoder configurations**, all compliant with the JPEG baseline standard.

---

## Final Remarks

This project implements a **true JPEG decoder operating directly on bitstreams**, rather than relying on existing libraries for coefficient extraction.

The ablation study provides insight into how different implementation choices affect decoding performance while preserving correctness.
