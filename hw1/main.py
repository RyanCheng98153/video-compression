import os
import cv2
import numpy as np
# a simple color transform implementation

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    # convert the image from BGR to RGB
    # due to OpenCV read image in BGR format
    # img = [h, w, [r, g, b]]

    # convert the image from BGR to RGB
    img = [
        [
            [r, g, b] 
            for b, g, r in row
        ] for row in img
    ]
    
    return img

def convert_YUV(img: np.ndarray) -> np.ndarray:
    # convert the image from RGB to YUV
    # img = [h, w, [r, g, b]]
    yuv = [
        [
            [
                0.299*r + 0.587*g + 0.114*b,          # Y
                -0.169*r - 0.331*g + 0.5*b + 128,     # U
                0.5*r - 0.419*g - 0.081*b + 128       # V
            ]
            for r, g, b in row
        ]
        for row in img
    ]
    
    # clip the values to [0, 255] and convert to uint8
    return np.clip(np.array(yuv), 0, 255).astype(np.uint8)

def convert_YCbCr(img: np.ndarray) -> np.ndarray:
    # convert the image from RGB to YCbCr
    # img = [h, w, [r, g, b]]
    ycbcr = [
        [
            [
                0.299*r + 0.587*g + 0.114*b,          # Y
                -0.168736*r - 0.331264*g + 0.5*b + 128,     # Cb
                0.5*r - 0.418688*g - 0.081312*b + 128       # Cr
            ]
            for r, g, b in row
        ]
        for row in img
    ]
    return np.clip(np.array(ycbcr), 0, 255).astype(np.uint8)

def main():
    img = cv2.imread("./lena.png")
    # The image is read in np.ndarray format with shape (h, w, [b, g, r])
    # We need to convert it to (h, w, [r, g, b])

    RGB_img = bgr_to_rgb(img)       # BGR to RGB
    YUV_img = convert_YUV(RGB_img)  # RGB to YUV
    YCbCr_img = convert_YCbCr(RGB_img)  # RGB to YCbCr

    # 建立輸出資料夾
    os.makedirs("./output", exist_ok=True)

    # RGB gray scale image
    # image = [h, w, [r, g, b]]
    R_img = [[ [pixel[0], pixel[0], pixel[0]] for pixel in row ] for row in RGB_img ] # R channel
    G_img = [[ [pixel[1], pixel[1], pixel[1]] for pixel in row ] for row in RGB_img ] # G channel
    B_img = [[ [pixel[2], pixel[2], pixel[2]] for pixel in row ] for row in RGB_img ] # B channel

    cv2.imwrite("./output/rgb_R.png", np.array(R_img))
    cv2.imwrite("./output/rgb_G.png", np.array(G_img))
    cv2.imwrite("./output/rgb_B.png", np.array(B_img))

    # YUV gray scale image
    # image = [h, w, [y, u, v]]
    yuv_Y_img = [[ [pixel[0], pixel[0], pixel[0]] for pixel in row ] for row in YUV_img ] # Y channel
    yuv_U_img = [[ [pixel[1], pixel[1], pixel[1]] for pixel in row ] for row in YUV_img ] # U channel
    yuv_V_img = [[ [pixel[2], pixel[2], pixel[2]] for pixel in row ] for row in YUV_img ] # V channel

    cv2.imwrite("./output/yuv_Y.png", np.array(yuv_Y_img))
    cv2.imwrite("./output/yuv_U.png", np.array(yuv_U_img))
    cv2.imwrite("./output/yuv_V.png", np.array(yuv_V_img))

    # YCbCr gray scale image
    # image = [h, w, [y, cb, cr]]
    ycbcr_Y_img = [[ [pixel[0], pixel[0], pixel[0]] for pixel in row ] for row in YCbCr_img ] # Y channel
    ycbcr_Cb_img = [[ [pixel[1], pixel[1], pixel[1]] for pixel in row ] for row in YCbCr_img ] # Cb channel
    ycbcr_Cr_img = [[ [pixel[2], pixel[2], pixel[2]] for pixel in row ] for row in YCbCr_img ] # Cr channel

    cv2.imwrite("./output/ycbcr_Y.png", np.array(ycbcr_Y_img))
    cv2.imwrite("./output/ycbcr_Cb.png", np.array(ycbcr_Cb_img))
    cv2.imwrite("./output/ycbcr_Cr.png", np.array(ycbcr_Cr_img))

    print("All images have been saved in ./output/")

if __name__ == "__main__":
    main()