"""
    Script responsible to read-in
    the input image to be counted
    and carry out the detection and
    couting processes.
"""

# Imports
import cv2
import argparse
from scrips import Detector

# Argument parser (CLA)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# Load the image and carry out
# some pre-processing, such as
# greyscale conversion, blurring
# thresholding
image = cv2.imread(args("image"))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blurred, 60)
