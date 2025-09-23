import cv2
# a simple color transform implementation

def bgr_to_rgb(func):
    # a simple and concise comment for this decorator
    """
    A decorator to convert image 
    from BGR to RGB and back to BGR
    due to OpenCV's default color format is BGR
    """
    def wrapper(img, *args, **kwargs):
        # convert the image from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # call the original function
        img = func(img, *args, **kwargs)
        # convert the image back to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    return wrapper

@bgr_to_rgb
def convert_YUV(img):
    # image is in RGB format
    # img = [h, w, [r, g, b]]
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r, g, b = img[i, j]
            # simple color transform
            img[i, j] = [
                0.299*r + 0.587*g + 0.114*b,        # Y
                -0.169*r - 0.331*g + 0.5*b + 128,   # U
                0.5*r - 0.419*g - 0.081*b + 128     # V
            ]

    return img

if __name__ == "__main__":
    img = cv2.imread("./lena.png")
    
    # RGB to YUV
    YUV_img = convert_YUV(img)

    print(YUV_img)
    
    # show the image
    cv2.imshow("lena", img)
    cv2.waitKey(0)