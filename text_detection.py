# USAGE
# python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.object_detection import non_max_suppression
from locality_aware_nms import nms_locality
import numpy as np
import argparse
import time
import cv2

zoom_f = 2
min_conf = 0.8

def init_args():
    """
    construct the argument parser and parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str,
                    help="path to input image")
    parser.add_argument("-c", "--min-confidence", type=float, default=0.5,
                    help="minimum probability required to inspect a region")
    parser.add_argument("--crop_params", nargs='+', type=int, default=[-1, -1, -1, -1],
                    help="crop parameters for input image")
    return parser.parse_args()

def get_nearest_multiple(val, max_val, factor):
    fac = val * 1.0/factor

    if fac != int(fac):
        if fac - int(fac)> 0:
            adj = (int(fac)+1)*factor
            val = adj if adj <=max_val else int(fac)*factor
    return int(val)

def adjust_cropargs(image, crop_params, factor):
    """
    adjust the crop parameters such that the cropped image's height and width is a factor of 'factor'
    :param image:
    :param crop_params: [x,y, width, height]
    :param factor: the height and with need to be a factor of this number
    :return: updated crop parameters
    """
    crop_x, crop_y, crop_w, crop_h = crop_params

    H, W = image.shape[:2]

    crop_w = get_nearest_multiple(crop_w, W - crop_x, factor)

    crop_h = get_nearest_multiple(crop_h, H - crop_y, factor)

    return [crop_x, crop_y, crop_w, crop_h]

def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_conf:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            #startX = int(endX - w)
            #startY = int(endY - h)

            #co-ordinates of the rotated rectangle
            poly = [
                (int(endX - (h * sin + w * cos)), int(endY + w * sin - h * cos)),
                (int(endX - h * sin), int(endY - h * cos)),
                (int(endX), int(endY)),
                (int(endX - w * cos), int(endY + w * sin))
            ]

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append(poly)
            confidences.append(scoresData[x])

    return (rects, confidences)

def drawRotatedBoundBox(boxes, image):
    for box in boxes:
        for idx in range(4):
            if idx == 3:
                cv2.line(image, (box[idx][0], box[idx][1]), (box[0][0], box[0][1]), (0, 0, 255), 2)
            else:
                cv2.line(image, (box[idx][0], box[idx][1]), (box[idx + 1][0], box[idx + 1][1]), (0, 0, 255), 2)

def get_text_boxes(image, W, H):
    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (rects, confidences) = decode_predictions(scores, geometry)
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    #boxes = non_max_suppression(np.array(rects), probs=confidences)
    boxes, conf = nms_locality(np.array(rects, dtype=np.float32), confidences)

    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    return boxes

def main():
    args = init_args()
    image = cv2.imread(args.image)
    if image is None:
        print("Image not found")

    crop_params = args.crop_params

    if crop_params[0] == -1:
        crop_params = [0, 0, image.shape[1], image.shape[0] ]
    crop_x, crop_y, width, height = adjust_cropargs(image, crop_params, 32)

    image = image[crop_y:crop_y + height, crop_x:crop_x + width]
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (width * zoom_f, height * zoom_f)
    #rW = origW / float(newW)
    #rH = origH / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH), cv2.INTER_AREA)
    (height, width) = image.shape[:2]
    boxes = get_text_boxes(image, width, height)

    drawRotatedBoundBox(boxes, image)
    cv2.imwrite('output.png', image)
    cv2.imshow("Text Detection", image)
    cv2.waitKey(0)

main()