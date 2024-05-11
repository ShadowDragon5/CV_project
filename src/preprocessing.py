# Set the stage
import cv2
import numpy as np
from bisect import bisect
# import sys

# %%
# size of score tag
np.array([679 - 651, 960 - 320])


# %%
# Return the score tag coordinates
# TODO: reduce resolution of output score tag, automate detect resolution of sample image and score tag coordinates
def extract_score_tag(frame):
    width, height, _ = frame.shape
    # Resolution of 'WSC\ sample.png'
    baseline_width, baseline_height = (720, 1280)

    # Relevant rang minus one pixel border due to blurring spill
    # Handpicked image coordinates of 'WSC\ sample.png'
    a_x, a_y = (651, 320)  # Upper left corner of score tag
    b_x, b_y = (679, 960)  # Lower right corner of score tag

    # Rescale coordinates to frame resolution
    r_x = width / baseline_width  # width scaling ratio
    r_y = height / baseline_height  # height scaling ratio
    a_x, a_y = (int(np.ceil(a_x * r_x)), int(np.ceil(a_y * r_y)))
    b_x, b_y = (int(np.floor(b_x * r_x)), int(np.floor(b_y * r_y)))

    # Extract score tag
    # print(a_x,a_y,b_x,b_y)
    score_tag = frame[a_x:b_x, a_y:b_y]

    # Visualize relevant range
    # cv2.imshow('score_tag',score_tag)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return score_tag


# %%
# Resolution agnostic comparaison of score tags
# TODO: reduce the amount of pixels checked, uniform sparce sampling
def compare_score_tag(baseline, sample):
    baseline_width, baseline_height, _ = baseline.shape
    sample_width, sample_height, _ = sample.shape

    # Compute smallest resolution
    width, height = (
        min(baseline_width, sample_width),
        min(baseline_height, sample_height),
    )
    # print(baseline_width, baseline_height)
    # print(sample_width, sample_height)
    # print(width, height)

    # Resize inputs
    baseline_small = cv2.resize(baseline, (height, width))
    sample_small = cv2.resize(sample, (height, width))

    # Compute abs diff and sum up as score
    diff = cv2.absdiff(baseline_small, sample_small)
    diff_score = np.sum(diff)
    # print(diff_score)

    # Tolerance on error (max allowed score)
    expected_average_pixel_error = 15 / 100  # Handpicked
    threshold = expected_average_pixel_error * (width * height)

    return diff_score < threshold


# %%
# Houhh line implementation
# original source: https://github.com/alyssaq/hough_transform
# TODO: numpy vectorized version of the code (faster implementation in same source)
def hough_lines(binary_img, rho_step=0.5, theta_step=0.5, num_lines=4):
    width, height = binary_img.shape
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90, 90 - theta_step, theta_step))
    diag_len = np.ceil(np.sqrt(width**2 + height**2))  # max_dist
    rhos = np.arange(-diag_len, diag_len, rho_step)

    # Cache some resuable values
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((len(rhos), len(thetas)))
    # (row, col) indexes to edges
    y_idxs, x_idxs = np.nonzero(binary_img)
    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(len(thetas)):
            # Calculate rho
            rho = x * cos_theta[t_idx] + y * sin_theta[t_idx]
            r_idx = bisect(
                rhos, rho, hi=len(rhos) - 1
            )  # which in rhos is the closest one
            accumulator[r_idx, t_idx] += 1

    # old_accumulator = np.copy(accumulator)

    # Non maxima supression
    # num_lines = 4 #Assumption on the number of lines we have
    lines = []
    for i in range(num_lines):
        rho_max_i, theta_max_i = np.unravel_index(
            np.argmax(accumulator), accumulator.shape
        )
        lines.append((rhos[rho_max_i], thetas[theta_max_i]))
        r_window = (
            10  # Assumption on the lines we want to detect having notably different rho
        )
        t_window = 10  # Lines we want to detect have different theta
        for r_w in range(-r_window, r_window + 1):
            r_w_mod = min(max((rho_max_i + r_w), 0), len(rhos))
            for t_w in range(-t_window, t_window + 1):
                t_w_mod = (theta_max_i + t_w) % len(thetas)
                # Supress maxima in neibourhood
                accumulator[r_w_mod, t_w_mod] = 0

    # accumulator = np.absolute(old_accumulator - accumulator)

    # Equalization and scaling for visualization
    # accumulator = np.floor(accumulator*255/(np.max(accumulator))).astype(np.uint8)
    # new_scale = np.flip(np.array(accumulator.shape) * 8)
    # acc_img = cv2.resize(accumulator,new_scale)
    # cv2.imshow('tmp', acc_img)

    return lines


# %%
# Function that detects the hardcoded green and brown colors
# and computes the edges where the colors touch. It then
# computes the most likely 4 straight lines that fit the edges
def table_edges_detection(frame_BGR):
    width, height, _ = frame_BGR.shape

    frame_HSV = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HSV)

    # Filter green color
    green_HSV_low = np.array([50, 200, 110])  # Hardcoded green HSV value
    green_HSV_high = np.array([70, 255, 210])
    frame_green = cv2.inRange(frame_HSV, green_HSV_low, green_HSV_high)

    # Filter brown color
    brown_HSV_low = np.array([0, 70, 35])  # Hardcoded brown HSV value
    brown_HSV_high = np.array([40, 210, 130])
    frame_brown_1 = cv2.inRange(frame_HSV, brown_HSV_low, brown_HSV_high)

    brown_HSV_low = np.array([150, 70, 35])  # Other brown
    brown_HSV_high = np.array([255, 210, 130])
    frame_brown_2 = cv2.inRange(frame_HSV, brown_HSV_low, brown_HSV_high)

    frame_brown = cv2.bitwise_or(frame_brown_1, frame_brown_2)

    # Noise filtering
    # frame_green = cv2.morphologyEx(frame_green, cv2.MORPH_OPEN, kernel)
    # frame_green = cv2.morphologyEx(frame_green, cv2.MORPH_CLOSE, kernel)

    # Edge detection between green and brown
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    green_dilate = cv2.dilate(frame_green, kernel)
    brown_dilate = cv2.dilate(frame_brown, kernel)
    green_brown_edges = cv2.bitwise_and(green_dilate, brown_dilate)

    # Edge detection using morphological gradient
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    # frame_brown = cv2.morphologyEx(frame_brown, cv2.MORPH_GRADIENT, kernel)

    lines = hough_lines(green_brown_edges)

    # Draw lines from Hough transform
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0]
            theta = lines[i][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(frame_BGR, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # frame -> frame_BGR
    # frame[:,:,0] = np.zeros((width,height))
    # frame[:,:,1] = frame_brown
    # frame[:,:,1] = cv2.bitwise_or(frame[:,:,1],green_brown_edges)
    return frame_BGR
