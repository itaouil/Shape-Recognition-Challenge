"""
    Script responsible to read-in
    the input image to be counted
    and carry out the processing,
    detection and counting logics.
"""

# Imports
import cv2
import argparse
import numpy as np
from scripts import ShapeRecognition

# Detector object
sr = ShapeRecognition()

# Counters
circles = 0
squares = 0
triangles = 0

# Argument parser (CLA)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# Load image passed from CLA
image = cv2.imread(args["image"])

# Pre-process image
(processed, ratio) = sr.process(image)

# Compute contours on the processed image
(_, cnts, _) = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Count objects
for c in cnts:
    # Compute contours center
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)

    # Detect the blob shape and
    # increase relative counters
    shape = sr.detect(c)
    if shape == "circle":
        circles += 1
    elif shape == "square":
        squares += 1
    else:
        triangles += 1

    # Resize contour to original ratio
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")

    # Draw contour with nearby text
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

print(str(squares), str(circles), str(triangles))

# # Display image
# cv2.imshow("Image", image)
# cv2.waitKey(0)
