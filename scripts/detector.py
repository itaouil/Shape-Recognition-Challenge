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

    def target(self, approximate):
        """
            Returns the target class
            of the blob based on the approximated contour.

            Arguments:
                approximate: Approx. contour

            Returns:
                class: Either triangle, circle, square
        """
        return {
            '3': "triangle",
            '4': "square"
        }[approximate]

    def detect(self, c):
        """
            The method uses the contour
            information of the blob which
            class it belongs to among the
            one possible.

            Arguments:
                c: Computed contour of the blob
        """
        # Compute perimeter of the contour
        perimeter = cv2.arcLength(c, True)

        # Approximate the contour curve
        # in order to obtain approximate
        # number of vertices given by the
        # intersection of the short lines
        approximate = cv2.approxPolyDP(c, 0.04 * perimeter, True)

        return target(approximate) if target(approximate) else "circle"
