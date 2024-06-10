import cv2
import numpy as np
from cv2.typing import MatLike

# local imports
from hough_line import hough_lines


def table_edges_detection(frame_BGR: MatLike) -> (MatLike, list):
    """
    Function that detects the hardcoded green and brown colors
    and computes the edges where the colors touch. It then
    computes the most likely 4 straight lines that fit the edges
    """
    width, height, _ = frame_BGR.shape

    frame_HSV = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HSV)

    # CV2 HSV ranges: Hue from 0 to 180, Staturation form 0 to 100, Value from 0 to 100

    # Filter green color
    # NOTE: Hard-coded green HSV value
    GREEN_HSV_LOW = np.array([50, 200, 50])
    GREEN_HSV_HIGH = np.array([70, 255, 220])
    frame_green = cv2.inRange(frame_HSV, GREEN_HSV_LOW, GREEN_HSV_HIGH)

    # Filter brown color
    # NOTE: Hard-coded brown HSV value
    BROWN_HSV_LOW = np.array([0, 0, 20])
    BROWN_HSV_HIGH = np.array([20, 200, 150])
    frame_brown_1 = cv2.inRange(frame_HSV, BROWN_HSV_LOW, BROWN_HSV_HIGH)

    # NOTE: Hard-coded another brown HSV value
    BROWN_HSV_LOW = np.array([120, 0, 20])
    BROWN_HSV_HIGH = np.array([180, 200, 150])
    frame_brown_2 = cv2.inRange(frame_HSV, BROWN_HSV_LOW, BROWN_HSV_HIGH)

    frame_brown = cv2.bitwise_or(frame_brown_1, frame_brown_2)

    # Noise filtering
    # frame_green = cv2.morphologyEx(frame_green, cv2.MORPH_OPEN, kernel)
    # frame_green = cv2.morphologyEx(frame_green, cv2.MORPH_CLOSE, kernel)

    # Dilate green and brown so that they overlap
    # NOTE: Hard-coded value
    KERNEL_SMALL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    brown_dilate = cv2.dilate(frame_brown, KERNEL_SMALL)
    green_dilate = cv2.dilate(frame_green, KERNEL_SMALL)

    # Remove some balls from the green playing area (red balls are too big of a blob)
    # NOTE: Hard-coded value
    KERNEL_BALL_SIZE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    green_close = cv2.morphologyEx(green_dilate, cv2.MORPH_CLOSE, KERNEL_BALL_SIZE)

    # Get a thin edge of the green playing area
    # NOTE: Hard-coded value
    KERNEL_VERY_SMALL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    green_gradient = cv2.morphologyEx(
        green_close, cv2.MORPH_GRADIENT, KERNEL_VERY_SMALL
    )

    # Intersect the green edge with the brown mask to obtain green/brow edges
    edges_green_brown = cv2.bitwise_and(green_gradient, brown_dilate)

    # # Draw edges in white
    # edges_negative = cv2.bitwise_not(edges_green_brown)
    # frame_BGR = cv2.bitwise_and(frame_BGR, frame_BGR, mask=edges_negative)
    # frame_BGR = frame_BGR + cv2.cvtColor(edges_green_brown, cv2.COLOR_GRAY2RGB)

    # Compute lines that fit the edges
    lines = hough_lines(edges_green_brown, num_lines=4)

    if lines is None:
        print("No lines found")
        return frame_BGR

    # Draw lines from Hough transform
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))

        cv2.line(frame_BGR, (x1, y1), (x2, y2), (0, 0, 255), 1)
        line = np.cross([x1, y1, 1], [x2, y2, 1])
        line = line / line[2]

    return frame_BGR, lines


def detect_baulk_line(frame: MatLike):
    """
    Function that detects the lines markings on the table
    frame: BGR image
    returns: baulk line in (rho, theta) format
    """
    # Filter white color
    white_HSL_low = np.array([9, 113, 2])  # Hardcoded white HSL value
    brown_HSL_high = np.array([13, 128, 10])
    frame_white = cv2.inRange(frame, white_HSL_low, brown_HSL_high)

    baulk_line = hough_lines(frame_white, num_lines=1)

    return baulk_line


def get_ball_centers(frame: MatLike) -> list:
    """
    frame: BGR image to detect the balls in
    returns: list of ball centers and their colors [(color, [x, y]) ...]
    """
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # NOTE: Hard-coded values
    ranges = {
        "yellow": [20, 35],
        "brown": [4, 23],
        "green": [71, 85],
        "blue": [85, 110],
    }
    KERNEL_BALL_SIZE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

    centers = {}
    for color, (lower, upper) in ranges.items():
        ball = cv2.inRange(img_hsv[:, :, 0], np.array([lower]), np.array([upper]))
        ball = cv2.erode(ball, KERNEL_BALL_SIZE)
        # get the average (middle) pixel x and y coordinate of the eroded ball
        centers[color] = np.mean(np.argwhere(ball != 0), axis=0, dtype=int)[::-1]

    return list(centers.items())
