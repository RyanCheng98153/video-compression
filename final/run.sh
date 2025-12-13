#!/bin/bash

PNG=images/lena.png
JPG=images/lena.jpg
OUT=results
mkdir -p ${OUT}

python png_to_jpg.py --png ${PNG} --jpg ${JPG}

echo "Running JPEG Decoder Ablation Study..."
echo "PNG: ${PNG}"
echo "JPG: ${JPG}"
echo "=============================="

# CSV Header
echo "YCbCr,IDCT,QTable,Time,PSNR,SSIM" > ${OUT}/results.csv

for YCBCR in formula table
do
  for IDCT in 2d two1d
  do
    for Q in 1 2
    do
      echo "----------------------------------"
      echo "YCbCr=${YCBCR}, IDCT=${IDCT}, QTable=${Q}"

      python main.py \
        --png ${PNG} \
        --jpg ${JPG} \
        --ycbcr ${YCBCR} \
        --idct ${IDCT} \
        --qtable ${Q} \
        | tee ${OUT}/log_${YCBCR}_${IDCT}_Q${Q}.txt

      # Extract metrics to CSV
      TIME=$(grep "Time" ${OUT}/log_${YCBCR}_${IDCT}_Q${Q}.txt | awk '{print $3}')
      PSNR=$(grep "PSNR" ${OUT}/log_${YCBCR}_${IDCT}_Q${Q}.txt | awk '{print $3}')
      SSIM=$(grep "SSIM" ${OUT}/log_${YCBCR}_${IDCT}_Q${Q}.txt | awk '{print $3}')

      echo "${YCBCR},${IDCT},${Q},${TIME},${PSNR},${SSIM}" >> ${OUT}/results.csv

    done
  done
done

echo "=============================="
echo "All experiments completed."
echo "Results saved to ${OUT}/results.csv"
