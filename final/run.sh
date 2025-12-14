#!/bin/bash

IMAGES=("lena")          # extend to ("lena" "tiger" "pizza")
IMG_DIR=images
OUT=results

mkdir -p ${OUT}

echo "Running JPEG Decoder Ablation Study (Multi-image, 10 runs)"
echo "========================================================="

# Global CSV header
echo "Image,YCbCr,IDCT,QTable,\
run_1,run_2,run_3,run_4,run_5,run_6,run_7,run_8,run_9,run_10,\
time_mean,time_std,time_total,PSNR,SSIM" \
> ${OUT}/results_all.csv


for IMG in "${IMAGES[@]}"
do
  PNG=${IMG_DIR}/${IMG}.png
  JPG=${IMG_DIR}/${IMG}.jpg
  IMG_OUT=${OUT}/${IMG}
  IMG_IMG_OUT=${IMG_OUT}/result_images

  mkdir -p ${IMG_OUT}
  mkdir -p ${IMG_IMG_OUT}

  echo "========================================================="
  echo "Processing image: ${IMG}"
  echo "PNG: ${PNG}"
  echo "JPG: ${JPG}"

  # PNG -> JPG
  python png_to_jpg.py --png ${PNG} --jpg ${JPG}

  # Per-image CSV header
  echo "YCbCr,IDCT,QTable,\
run_1,run_2,run_3,run_4,run_5,run_6,run_7,run_8,run_9,run_10,\
time_mean,time_std,time_total,PSNR,SSIM" \
  > ${IMG_OUT}/results.csv


  for YCBCR in formula table
  do
    for IDCT in 2d two1d
    do
      for Q in 1 2
      do
        echo "---------------------------------------------------------"
        echo "Image=${IMG}, YCbCr=${YCBCR}, IDCT=${IDCT}, QTable=${Q}"

        LOG=${IMG_OUT}/log_${YCBCR}_${IDCT}_Q${Q}.txt

        python main.py \
          --png ${PNG} \
          --jpg ${JPG} \
          --ycbcr ${YCBCR} \
          --idct ${IDCT} \
          --qtable ${Q} \
          --out_img_dir ${IMG_IMG_OUT} \
          | tee ${LOG}

        # -------------------------
        # Extract values from log
        # -------------------------
        RUNS=$(grep "Run times" ${LOG} | cut -d':' -f2 | tr -d ' ')
        TIME_MEAN=$(grep "Time mean" ${LOG} | awk '{print $4}')
        TIME_STD=$(grep "Time std" ${LOG} | awk '{print $4}')
        TIME_TOTAL=$(grep "Time total" ${LOG} | awk '{print $4}')
        PSNR=$(grep "PSNR" ${LOG} | awk '{print $3}')
        SSIM=$(grep "SSIM" ${LOG} | awk '{print $3}')

        # Per-image CSV
        echo "${YCBCR},${IDCT},${Q},${RUNS},${TIME_MEAN},${TIME_STD},${TIME_TOTAL},${PSNR},${SSIM}" \
          >> ${IMG_OUT}/results.csv

        # Global CSV
        echo "${IMG},${YCBCR},${IDCT},${Q},${RUNS},${TIME_MEAN},${TIME_STD},${TIME_TOTAL},${PSNR},${SSIM}" \
          >> ${OUT}/results_all.csv

      done
    done
  done
done

echo "========================================================="
echo "All experiments completed."
echo "Per-image results : ${OUT}/<image>/results.csv"
echo "Overall results   : ${OUT}/results_all.csv"
echo "Images saved in   : ${OUT}/<image>/result_images/"
