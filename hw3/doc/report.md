# Homework 3 Report: Motion Estimation
**Student ID:** [314554025]  
**Name:** [鄭睿宏]  
**Project Link:** https://github.com/RyanCheng98153/video-compression/tree/main/hw3

#### 1. Introduction

In this homework, I implemented **Full Search (FS)** and **Three-Step Search (TSS)** motion estimation algorithms for video compression. The block size is set to \(8\times8\), and integer precision is used. Motion compensation (MC) reconstructs the current frame using the estimated motion vectors, and the residual is computed as the difference between the reconstructed and original frames. 

We evaluate the quality of motion estimation using **PSNR** and compare the **runtime** for different search ranges (\(\pm8, \pm16, \pm32\)).

#### 2. Motion Estimation Methods

**Full Search (FS)** exhaustively searches all candidate blocks within the specified search range. For each block in the current frame \(C\), it finds the displacement \((\Delta x, \Delta y)\) that minimizes the **Sum of Absolute Differences (SAD)** with the reference frame \(R\):

\[
(\Delta x, \Delta y) = \arg\min_{(dx,dy) \in [-R,R]^2} \sum_{i=0}^{B-1} \sum_{j=0}^{B-1} | C(i,j) - R(i+dx, j+dy) |
\]

where \(B\) is the block size (8 in this case), and \(R\) is the search range.  
While FS guarantees finding the optimal block match, its computational complexity grows quadratically with the search range:

\[
O(B^2 \cdot (2R+1)^2 \cdot N_\text{blocks})
\]

where \(N_\text{blocks}\) is the total number of blocks.

**Three-Step Search (TSS)** is a fast search algorithm that reduces computation by iteratively refining the search in three steps. At each step, it checks nine candidate positions around the current best match with a decreasing step size:

\[
\text{step}_0 = 2^{\lfloor \log_2 R \rfloor}, \quad
\text{step}_{k+1} = \max(1, \text{step}_k / 2), \quad k = 0,1,2
\]

TSS significantly reduces the number of block comparisons, achieving much lower runtime at the cost of slightly suboptimal motion vectors in some cases.

#### 3. Results

- Reference and current grayscale images:
- one_gray.png
<img src="../figures/full/full_r8_recon.png" width="250" height="250"> &nbsp; &nbsp; <!-- Example reference -->
- two_gray.png
<img src="../figures/tss/tss_r8_recon.png" width="250" height="250">


<!-- Page break -->
<div style="break-after: page; page-break-after: always;"></div>

**Full Search Results (FS)**  

| Search Range | PSNR (dB) | Time (s) | Reconstructed Image | Residual Image |
|--------------|-----------|----------|------------------|----------------|
| ±8           | 22.95     | 1.138    | ![full_r8](../figures/full/full_r8_recon.png) | ![residual](../figures/full/full_r8_residual_vis.png) |
| ±16          | 25.51     | 4.240    | ![full_r16](../figures/full/full_r16_recon.png) | ![residual](../figures/full/full_r16_residual_vis.png) |
| ±32          | 29.10     | 16.546   | ![full_r32](../figures/full/full_r32_recon.png) | ![residual](../figures/full/full_r32_residual_vis.png) |


<!-- Page break -->
<div style="break-after: page; page-break-after: always;"></div>

**Three-Step Search Results (TSS)**  

| Search Range | PSNR (dB) | Time (s) | Reconstructed Image | Residual Image |
|--------------|-----------|----------|------------------|----------------|
| ±8           | 21.92     | 0.127    | ![tss_r8](../figures/tss/tss_r8_recon.png) | ![residual](../figures/tss/tss_r8_residual_vis.png) |
| ±16          | 21.91     | 0.128    | ![tss_r16](../figures/tss/tss_r16_recon.png) | ![residual](../figures/tss/tss_r16_residual_vis.png) |
| ±32          | 21.74     | 0.124    | ![tss_r32](../figures/tss/tss_r32_recon.png) | ![residual](../figures/tss/tss_r32_residual_vis.png) |


<!-- Page break -->
<div style="break-after: page; page-break-after: always;"></div>

#### 4. Observations

- **Accuracy vs Range:**  
  Full Search achieves higher PSNR as the search range increases because it explores more candidate blocks and finds better matches. TSS shows almost constant PSNR across ranges because it searches fewer positions, potentially missing the optimal block for larger ranges.

- **Runtime:**  
  TSS is significantly faster than Full Search. For example, at range ±32, FS takes 16.5 s, whereas TSS completes in 0.124 s—more than 100× speed-up.

- **Residual Analysis:**  
  Residual images for FS show lower intensity (smaller errors) compared to TSS, indicating more accurate block prediction.

#### 5. Conclusion

Full Search guarantees optimal motion estimation but is computationally expensive. Three-Step Search provides a **fast approximation** with drastically reduced runtime at the cost of some quality loss. The trade-off between accuracy (PSNR) and efficiency (runtime) should guide the choice of algorithm depending on application needs.
