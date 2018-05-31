"""
    Script responsible to read-in
    the input image to be counted
    and carry out the detection and
    couting processes.
"""

# Imports
import cv2
import imutils
import argparse
import numpy as np
from scripts import Detector

# Argument parser (CLA)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# Load the image and resize it
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# Image pre-processing
grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(grey, 5)
thresh = cv2.adaptiveThreshold(blurred,
                               255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY,
                               11,
                               25)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(thresh, kernel, iterations = 1)
edges = cv2.Canny(erosion, 100, 200)

# Compute contours on the binarized
# image (now easier thanks to gradient
# differences)
cv2.imshow("thresh", thresh)
cv2.imshow("edges", edges)
cv2.imshow("erosion", erosion)
cv2.waitKey(0)

(_, cnts, _) = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
d = Detector()

print(len(cnts))

# loop over the contours
for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    print(M)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = d.detect(c)

    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
