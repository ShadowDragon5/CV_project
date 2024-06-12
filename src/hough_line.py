from bisect import bisect

import numpy as np


# TODO: numpy vectorized version of the code (faster implementation in same source)
def hough_lines(binary_img, rho_step=0.5, theta_step=0.5, num_lines=4):
    """
    Hough line implementation
    original source: https://github.com/alyssaq/hough_transform
    """
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
            20  # Assumption on the lines we want to detect having notably different rho
        )
        t_window = 20  # Lines we want to detect have different theta
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
