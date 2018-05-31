"""
    Class that given the properties of
    the blob (mainly countours) assigns
    a class to it, which can either be
    a circle, triangle or square.
"""

# Imports
import cv2
import imutils
import numpy as np

class ShapeRecognition:

    def __init__(self):
        """
            Constructor.

            Literally None.
        """
        pass

    def process(self, image):
        """
            The method applies different image
            pre-processing algorithms such as
            thresholding, blurring, edge detection
            and Laplacian before further computations.

            Arguments:
                MAT image: OpenCV image

            Returns:
                MAT image: Pre-processed image
        """
        # Resize the image for better countour fitting
        # and store ratio gap
        resized = imutils.resize(image, width=300)
        ratio = image.shape[0] / float(resized.shape[0])

        # Convert the resized image into greyscale
        grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Blur the image to remove noise
        blurred = cv2.medianBlur(grey, 5)

        # Otsu threshold for the win
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform erosion to make edges bigger
        # following by Canny edge detection for
        # contour computation
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations = 1)
        dilate = cv2.dilate(erosion, kernel, iterations = 1)
        edges = cv2.Canny(dilate, 30, 120)

        # TODO: Add laplacian to avoid double edge detection

        cv2.imshow("grey", grey)
        cv2.imshow("blurred", blurred)
        cv2.imshow("thresh", thresh)
        cv2.imshow("edges", edges)
        cv2.imshow("dilate", dilate)
        cv2.imshow("erosion", erosion)
        cv2.waitKey(0)

        return edges, ratio

    def detect(self, c):
        """
            The method uses the contour
            information of the blob which
            class it belongs to among the
            one possible.

            Arguments:
                c: Computed contour of the blob

            Returns:
                class: Either triangle, circle or square
        """
        # Compute perimeter of the contour
        perimeter = cv2.arcLength(c, True)

        # Approximate the contour curve
        # in order to obtain approximate
        # number of vertices given by the
        # intersection of the short lines
        approximate = cv2.approxPolyDP(c, 0.04 * perimeter, True)

        # Return blob's relative
        # target class based on
        # number of approximated
        # vertices
        if len(approximate) == 3:
            return "triangle"

        elif len(approximate) == 4:
            return "square"

        else:
            return "circle"
