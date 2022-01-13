import glob
import cv2
import numpy as np

def get_bbox_from_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2 = imutils.grab_contours(contours)

    bbox = np.empty([len(contours2), 4], dtype="int32")

    for i, contour in enumerate(contours2):
        x = contour[0][0][0]
        y = contour[0][0][1]

        h = contour[2][0][1] - contour[0][0][1]
        w = contour[2][0][0] - contour[0][0][0]

        # print(contour, "-->", x, y, h, w)
        bbox[i][0] = x
        bbox[i][1] = y
        bbox[i][2] = w
        bbox[i][3] = h

    return bbox


def bounding_rect_to_corners(rect):
    corners = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
    return corners


def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


cv_img = []


left_ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')

for image in glob.glob("C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/MOJE/data/ears/tempTest2/*.png"):
    img = cv2.imread(image)
    boxes = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left_ear = left_ear_cascade.detectMultiScale(gray, 1.07, 2)
    right_ear = right_ear_cascade.detectMultiScale(gray, 1.07, 2)

    for (x, y, w, h) in left_ear:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # boxes.append([x, y, w, h])
        print(x, y, w, h)

    for (x, y, w, h) in right_ear:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        # boxes.append([x, y, w, h])
        print(x, y, w, h)
    #
    # number = img[-8:-4]
    # maskloc = r"C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/IBB-Ear-Detection/testannot_rect/" + number + ".png"
    # mask = cv2.imread(maskloc)
    # mask_boxes = get_bbox_from_mask(mask)
    # mask_checked = [0] * len(mask_boxes)


    filename = "../detected.png"
    cv2.imwrite(filename, img)
    cv2.imshow('Ear Detector', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


