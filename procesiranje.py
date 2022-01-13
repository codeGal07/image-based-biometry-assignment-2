import glob
import cv2
import numpy as np
cv_img = []


for img in glob.glob("C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/MOJE/data/ears/temp/*.png"):
    im = cv2.imread(img)

    #sharpening:
    sharpen = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    im = cv2.filter2D(src=im, ddepth=-1, kernel=sharpen)

    #denoising
    im = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 7, 21)

    im = cv2.addWeighted(im, 0.8, im, 0, 1)

    cv_img.append(im)
    cv2.imwrite('C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/MOJE/data/ears/temp2/'+  img[-8:], im)


