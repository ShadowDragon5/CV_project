import cv2
import numpy as np
from cv2.typing import MatLike
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
    GREEN_HSV_LOW = np.array([55, 100, 50])
    GREEN_HSV_HIGH = np.array([65, 255, 220])
    frame_green = cv2.inRange(frame_HSV, GREEN_HSV_LOW, GREEN_HSV_HIGH)

    # Filter brown color
    # NOTE: Hard-coded brown HSV value
    BROWN_HSV_LOW = np.array([0, 0, 20])
    BROWN_HSV_HIGH = np.array([5, 200, 150])
    frame_brown_1 = cv2.inRange(frame_HSV, BROWN_HSV_LOW, BROWN_HSV_HIGH)

    # NOTE: Hard-coded another brown HSV value
    BROWN_HSV_LOW = np.array([130, 0, 20])
    BROWN_HSV_HIGH = np.array([180, 200, 150])
    frame_brown_2 = cv2.inRange(frame_HSV, BROWN_HSV_LOW, BROWN_HSV_HIGH)

    frame_brown = cv2.bitwise_or(frame_brown_1, frame_brown_2)

    # Noise filtering
    # frame_green = cv2.morphologyEx(frame_green, cv2.MORPH_OPEN, kernel)
    # frame_green = cv2.morphologyEx(frame_green, cv2.MORPH_CLOSE, kernel)

    # Dilate green and brown so that they overlap
    # NOTE: Hard-coded value
    KERNEL_SMALL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    brown_dilate = cv2.dilate(frame_brown, KERNEL_SMALL)
    KERNEL_VERY_SMALL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    green_dilate = cv2.dilate(frame_green, KERNEL_VERY_SMALL)

    # Remove some balls from the green playing area (red balls are too big of a blob)
    # NOTE: Hard-coded value
    # KERNEL_BALL_SIZE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # green_close = cv2.morphologyEx(green_dilate, cv2.MORPH_CLOSE, KERNEL_BALL_SIZE)

    # Get a thin edge of the green playing area
    # NOTE: Hard-coded value
    KERNEL_VERY_SMALL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    green_gradient = cv2.morphologyEx(
        green_dilate, cv2.MORPH_GRADIENT, KERNEL_VERY_SMALL
    )

    # Intersect the green edge with the brown mask to obtain green/brow edges
    edges_green_brown = cv2.bitwise_and(green_gradient, brown_dilate)

    # # Draw edges in white
    # edges_negative = cv2.bitwise_not(edges_green_brown)
    # frame_BGR = cv2.bitwise_and(frame_BGR, frame_BGR, mask=edges_negative)
    # frame_BGR = frame_BGR + cv2.cvtColor(edges_green_brown, cv2.COLOR_GRAY2RGB)

    # Compute lines that fit the edges
    lines = hough_lines(edges_green_brown, num_lines=4)

    # NOTE: order is important
    l_top = lines[0]
    l_bot = lines[1]
    l_rig = lines[2]
    l_lef = lines[3]

    def A_b(rho1, theta1, rho2, theta2):
        A = np.array(
            [
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)],
            ]
        )
        b = np.array([[rho1, rho2]]).T
        return A, b

    # find the intersection points
    # $\begin{bmatrix} \cos\theta_1 & \sin\theta_1 \\ \cos\theta_2 & \sin\theta_2 \end{bmatrix} \cdot \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} \rho_1 \\ \rho_2 \end{bmatrix}$
    top_left = np.linalg.solve(*A_b(*l_top, *l_lef)).T.squeeze()
    top_right = np.linalg.solve(*A_b(*l_top, *l_rig)).T.squeeze()
    bottom_left = np.linalg.solve(*A_b(*l_bot, *l_lef)).T.squeeze()
    bottom_right = np.linalg.solve(*A_b(*l_bot, *l_rig)).T.squeeze()

    # # Draw lines from Hough transform
    # for rho, theta in lines:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 10000 * (-b))
    #     y1 = int(y0 + 10000 * (a))
    #     x2 = int(x0 - 10000 * (-b))
    #     y2 = int(y0 - 10000 * (a))
    #     cv2.line(frame_BGR, (x1, y1), (x2, y2), (0, 0, 255), 1)
    #     img_show(frame_BGR)

    # return frame_BGR, lines
    return np.array(
        [[top_left, top_right, bottom_right, bottom_left]],
        dtype=np.int32,
    )


def detect_baulk_line(frame: MatLike) -> tuple:
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

    return baulk_line[0]


def get_ball_centers(frame: MatLike) -> dict:
    """
    frame: BGR image to detect the balls in
    returns: dictionary of ball colors and their centers {color: [x, y], ...}
    """
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # NOTE: Hard-coded values
    ranges = {
        "yellow": [20, 35],
        "brown": [15, 23],
        "green": [71, 85],
        "blue": [85, 110],
    }
    KERNEL_BALL_SIZE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

    centers = {}
    for color, (lower, upper) in ranges.items():
        ball = cv2.inRange(img_hsv[:, :, 0], np.array([lower]), np.array([upper]))
        eroded = cv2.erode(ball, KERNEL_BALL_SIZE)
        # get the average (middle) pixel x and y coordinate of the eroded ball
        centers[color] = np.mean(np.argwhere(eroded != 0), axis=0, dtype=int)[::-1]

    return centers
