import cv2
# a simple color transform implementation

if __name__ == "__main__":
    img = cv2.imread("data/lena.png")
    print(img.shape)
    