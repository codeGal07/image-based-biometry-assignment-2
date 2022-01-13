import cv2
import numpy as np
import glob
import random
import imutils


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


if __name__ == "__main__":
    # Load Yolo
    net = cv2.dnn.readNet("yolov3_custom_4000.weights", "yolov3_testing.cfg")

    # Name custom object
    classes = ["Ear"]
    images_path = glob.glob("C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/MOJE/data/ears/temp2/*.png")
    # print(images_path)
    # print(net.getUnconnectedOutLayers())

    x = net.getUnconnectedOutLayers()
    nov = []
    for i in range(0, x.size):
        nov.append([x[i]])

    layer_names = net.getLayerNames()
    # print(layer_names)
    output_layers = [layer_names[i[0] - 1] for i in nov]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    avgIOU = 0

    # Insert here the path of your images
    # random.shuffle(images_path)
    # loop through all the images
    for img_path in images_path:
        # Loading image
        img = cv2.imread(img_path)
        # img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    # print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # print(indexes)
        number = img_path[-8:-4]

        maskloc = r"C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/IBB-Ear-Detection/testannot_rect/" + number + ".png"
        mask = cv2.imread(maskloc)
        mask_boxes = get_bbox_from_mask(mask)
        mask_checked = [0] * len(mask_boxes)


        imgIOU = 0

        num_boxes = 0

        for i in range(len(boxes)):
            if i in indexes:
                num_boxes += 1
                b = bounding_rect_to_corners(boxes[i])

                bestIOU = 0
                # ker ne vem kero uho bo gledal in je best tist za katerega se pač gre
                for j in range(len(mask_boxes)):
                    m = bounding_rect_to_corners(mask_boxes[j])
                    iou = intersection_over_union(b, m)

                    if (iou > bestIOU):
                        bestIOU = iou
                    if (iou > mask_checked[j]):
                        mask_checked[j] = iou

                #sestevki za povprecje za best
                imgIOU = imgIOU + bestIOU
                # print("best-iou: " + str(bestIOU))

        negatives = 0
        #koliko je takšnih, ki jih ni zaznalo pa v resnici so
        for mc in mask_checked:
            if mc == 0:
                negatives += 1
        # print("negatives: " + str(negatives))

        if (imgIOU != 0):
            imgIOU = imgIOU / (num_boxes + negatives)

        # print("final-iou: " + str(imgIOU))
        avgIOU += imgIOU

    avgIOU /= len(images_path)
    print("AVERAGE IOU: " + str(avgIOU))

    cv2.destroyAllWindows()