#!/bin/bash

IMAGES=("lena")
IMG_DIR=images
OUT=results

mkdir -p ${OUT}

echo "Running JPEG Decoder Ablation Study (Multi-image)"
echo "==============================================="

# 全部圖片的總 CSV
echo "Image,YCbCr,IDCT,QTable,Time,PSNR,SSIM" > ${OUT}/results_all.csv

for IMG in "${IMAGES[@]}"
do
  PNG=${IMG_DIR}/${IMG}.png
  JPG=${IMG_DIR}/${IMG}.jpg
  IMG_OUT=${OUT}/${IMG}

  mkdir -p ${IMG_OUT}

  echo "==============================================="
  echo "Processing image: ${IMG}"
  echo "PNG: ${PNG}"
  echo "JPG: ${JPG}"

  # PNG -> JPG
  python png_to_jpg.py --png ${PNG} --jpg ${JPG}

  # 每張圖自己的 CSV
  echo "YCbCr,IDCT,QTable,Time,PSNR,SSIM" > ${IMG_OUT}/results.csv

  for YCBCR in formula table
  do
    for IDCT in 2d two1d
    do
      for Q in 1 2
      do
        echo "----------------------------------"
        echo "Image=${IMG}, YCbCr=${YCBCR}, IDCT=${IDCT}, QTable=${Q}"

        LOG=${IMG_OUT}/log_${YCBCR}_${IDCT}_Q${Q}.txt

        python main.py \
          --png ${PNG} \
          --jpg ${JPG} \
          --ycbcr ${YCBCR} \
          --idct ${IDCT} \
          --qtable ${Q} \
          | tee ${LOG}

        # Extract metrics
        TIME=$(grep "Time" ${LOG} | awk '{print $3}')
        PSNR=$(grep "PSNR" ${LOG} | awk '{print $3}')
        SSIM=$(grep "SSIM" ${LOG} | awk '{print $3}')

        # 寫入單張圖 CSV
        echo "${YCBCR},${IDCT},${Q},${TIME},${PSNR},${SSIM}" >> ${IMG_OUT}/results.csv

        # 寫入總表 CSV
        echo "${IMG},${YCBCR},${IDCT},${Q},${TIME},${PSNR},${SSIM}" >> ${OUT}/results_all.csv

      done
    done
  done
done

echo "==============================================="
echo "All experiments completed."
echo "Per-image results: ${OUT}/<image>/results.csv"
echo "Overall results  : ${OUT}/results_all.csv"
