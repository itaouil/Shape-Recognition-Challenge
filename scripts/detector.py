"""
    Class that given the properties of
    the blob (mainly countours) assigns
    a class to it, which can either be
    a circle, triangle or square.
"""

# Imports
import cv2

class Detector:

    def __init__(self):
        """
            Constructor.

            No need for it in this occasion
        """
        pass

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

        if len(approximate) == 3:
            return "triangle"

        elif len(approximate) == 4:
            return "square"

        else:
            return "circle"
