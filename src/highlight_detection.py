import itertools

import cv2
import numpy as np
from utils import img_show

# NOTE: Hard-coded vaues:
# Colors of the balls
COLORS = ["blue", "red", "pink", "brown", "yellow", "green", "white", "black"]
COLOR_STR_TO_BGR = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "pink": (130, 130, 255),
    "green": (0, 255, 0),
    "black": (50, 50, 50),
    "brown": (20, 70, 130),
    "yellow": (0, 190, 250),
}
# Color of the blue ball
MAX_BLUE = (25, 255, 255)
MIN_BLUE = (10, 0, 30)
# Color of the red balls
MAX_RED = (130, 255, 190)
MIN_RED = (110, 170, 0)
# Color of the pink ball
MAX_PINK = (130, 255, 255)
MIN_PINK = (110, 0, 200)
# Color of the brown ball
MAX_BROWN = (110, 255, 150)
MIN_BROWN = (90, 0, 0)
# Color of the yellow ball
MAX_YELLOW = (100, 255, 255)
MIN_YELLOW = (90, 0, 150)
# Color of the green ball
MAX_GREEN = (50, 255, 255)
MIN_GREEN = (30, 0, 0)
# Color of the white ball
MAX_WHITE = (90, 255, 255)
MIN_WHITE = (70, 0, 150)
# Color of the black ball
MAX_BLACK = (255, 255, 50)
MIN_BLACK = (0, 1, 0)
# Color of the highlight/specular
MIN_HIGHLIGHT = (0, 0, 245)
# This value is used to detect the larger specular area so is more tolerant
MAX_HIGHLIGHT_REGION = (255, 255, 255)
MAX_HIGHLIGHT = (255, 120, 255)
# Color of the highlight/specular on different balls:
MIN_HIGHLIGHT_BLACK = (0, 0, 145)
MAX_HIGHLIGHT_YELLOW = (255, 200, 255)
MAX_HIGHLIGHT_PINK = (255, 70, 255)
# Number of Iterations to erode the initial ball masks
ERODE_ITERATIONS = 2
ERODE_ITERATIONS_BLACK = 4
# Number of Iterations to dilate the initial ball masks
DILATE_ITERATIONS = 2


def get_neighbours(point):
    """
    Returns the neighbour coordinates of a pixel ignoring diagonals
    """
    x, y = point
    return [[x, y], [x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]


def merge_groups(groups, ignore_groups):
    """
    Takes a list of lists containing coordinates of pixels.
    These lists are called groups.
    Merges 2 groups if they contain neighbouring pixels.
    """
    ignore_groups = []  # ignore already processed groups to improve performance
    for current_index, current_group in enumerate(groups):
        if current_index in ignore_groups:
            continue
        for point in current_group:
            neighbours = get_neighbours(point)
            for index, group in enumerate(groups):
                if index == current_index:
                    continue
                # check for every pixel if it has neighbours that belong to another group
                exists = any(neighbour in group for neighbour in neighbours)
                if exists:
                    # merge the 2 groups
                    groups[index] = groups[index] + groups[current_index]
                    groups.pop(current_index)
                    # print("merged", index, current_index, "=", len(groups))
                    return groups, True, ignore_groups
        ignore_groups += [current_index]
    return groups, False, ignore_groups


def get_groups(mask, n=1):
    """
    Given a mask, returns the n pargest patches/groups of pixels with value 255.
    """
    # NOTE: Hard-coded value of the mask's value
    candidates = np.where(mask == 255)

    groups = []
    for px, py in zip(candidates[0], candidates[1]):
        added = False
        neighbours = get_neighbours((px, py))
        # check if the pixel belongs to an already created group
        for index, group in enumerate(groups):
            exists = any(elem in group for elem in neighbours)
            if exists:
                groups[index] = groups[index] + [[px, py]]
                added = True
        if not added:  # create new group
            groups += [[[px, py]]]

    # merge neighbouring groups, this is necessary as the previous step may have one patch into multiple groups
    ignore_groups = []
    merged = True
    while merged:  # and len(groups)>n: # TODO is this robust without the n (if there are less than 15 red balls)?
        groups, merged, ignore_groups = merge_groups(groups, ignore_groups)
    groups = sorted(groups, key=len, reverse=True)
    return groups


def matches_window(frame, lower_mask, color_min_upper, color_max_upper):
    """
    Checks for every pixel in a given mask if the above neighbour is withing a specified range.
    Returns the coordinates of the upper pixels of such matchings.
    """
    lower = lower_mask  # cv2.inRange(frame, color_min_lower, color_max_lower)
    upper = cv2.inRange(frame, color_min_upper, color_max_upper)

    lower_matched_x, lower_matched_y = np.where(lower == 255)

    matchings = []
    for lower_candidate_x, lower_candidate_y in zip(lower_matched_x, lower_matched_y):
        if lower[lower_candidate_x, lower_candidate_y] > 0:
            if upper[lower_candidate_x - 1, lower_candidate_y] == 255:
                matchings.append((lower_candidate_x - 1, lower_candidate_y))
        else:
            print("image window out of bounds")
    return matchings


def morphological_filter(group, dil_iter=0, ero_iter=0):
    # remove duplicates (uncaught bug)
    group.sort()
    group = list(group for group, _ in itertools.groupby(group))

    # Area filter

    # create zero's matrix as background
    ball_template = np.zeros((4 * dil_iter + 20, 4 * dil_iter + 20))

    # insert template ball
    ball_template[
        2 * dil_iter : 2 * dil_iter + 20, 2 * dil_iter : 2 * dil_iter + 20
    ] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    ball_template = cv2.erode(ball_template, None, iterations=ero_iter)
    ball_template = cv2.dilate(ball_template, None, iterations=dil_iter + ero_iter)
    # img_show(ball_template*255)

    area_ball = np.sum(ball_template)
    # print("Area", area_ball * 1.2, len(group))
    if len(group) > area_ball * 1.2:
        tmp = np.zeros((720, 1280))
        for [x, y] in group:
            if tmp[x, y] == 0:
                tmp[x, y] = 255
            else:
                print("PANIK")
        img_show(tmp)
        # print("Area fail")
        return False

    # Bounding box (orthogonal diameter)
    diameter_ball = 20 + 2 * dil_iter
    group_x = [pixel[0] for pixel in group]
    group_y = [pixel[1] for pixel in group]
    diameter_x = np.max(group_x) - np.min(group_x)
    diameter_y = np.max(group_y) - np.min(group_y)
    diameter_group = max(diameter_x, diameter_y)
    if diameter_group > diameter_ball * 1.2:
        print("Diameter fail")
        return False
    # print("Diameter", diameter_ball, diameter_group)

    return True


def visualize_points(image, points, color):
    """
    Takes an image and paints some points in specified color on top
    """
    image = image.copy()
    image = np.zeros((image.shape[0], image.shape[1], 3))  # .astype(int)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y in points:
        image[x, y] = color
    return image


def detect_highlight(frame, version="max"):
    """
    Given a frame, returns the coordinates of the highlights/speculars on all balls.
    Version="max": returns the lower edge of the highlight
    Version="mean": returns the middle of the highlight
    Return structure:
    [(color_as_string, (x_coord, y_coord), mask_of_highlight), ...]
    for red, the second element is a list of coordinates, as there may be multiple red balls:
    [(x_coord_1, y_coord_1), (x_coord_2, y_coord_2), ...]
    """
    # convert frame to HSV for easier color detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    mask_blue = cv2.inRange(hsv_frame, MIN_BLUE, MAX_BLUE)
    mask_red = cv2.inRange(hsv_frame, MIN_RED, MAX_RED)
    mask_pink = cv2.inRange(hsv_frame, MIN_PINK, MAX_PINK)
    mask_brown = cv2.inRange(hsv_frame, MIN_BROWN, MAX_BROWN)
    mask_yellow = cv2.inRange(hsv_frame, MIN_YELLOW, MAX_YELLOW)
    mask_green = cv2.inRange(hsv_frame, MIN_GREEN, MAX_GREEN)
    mask_white = cv2.inRange(hsv_frame, MIN_WHITE, MAX_WHITE)
    mask_black = cv2.inRange(hsv_frame, MIN_BLACK, MAX_BLACK)
    # img_show(mask_black)

    raw_masks = [
        mask_blue,
        mask_red,
        mask_pink,
        mask_brown,
        mask_yellow,
        mask_green,
        mask_white,
        mask_black,
    ]
    masks = []

    # Dilate all masks and get largest patch:
    for index, mask in enumerate(raw_masks):
        erode_iterations = ERODE_ITERATIONS
        dilate_iterations = DILATE_ITERATIONS
        if index == 1:  # red
            masks.append(mask)
            continue  # Not parsing red as there are multiple red balls
        if index == 7:  # black
            erode_iterations = ERODE_ITERATIONS_BLACK
        eroded_mask = cv2.erode(mask, None, iterations=erode_iterations)
        dilated_mask = cv2.dilate(
            eroded_mask, None, iterations=dilate_iterations + erode_iterations
        )

        groups = get_groups(dilated_mask, dilate_iterations - erode_iterations)

        # Decapsulate
        # Filter groups
        groups = [
            group
            for group in groups
            if morphological_filter(group, dilate_iterations, erode_iterations)
        ]
        if len(groups) == 0:
            print("No ball found")
            continue
        largest_group = groups[0]
        new_mask = np.zeros(mask.shape)
        for point in largest_group:
            new_mask[point[0], point[1]] = 255
        masks.append(new_mask)

    highlight_masks = []

    for mask in masks:
        matchings = matches_window(
            hsv_frame, mask, MIN_HIGHLIGHT_BLACK, MAX_HIGHLIGHT_REGION
        )
        matchings_visualized = visualize_points(mask, matchings, (0, 0, 255))
        highlight_masks.append(matchings_visualized)
        # cv2.imshow("matchings", matchings_visualized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    highlight_masks_diluted = []
    for index, highlight in enumerate(highlight_masks):
        eroded_mask = cv2.erode(highlight, None, iterations=0)
        dilated_mask = cv2.dilate(eroded_mask, None, iterations=4)
        highlight_masks_diluted.append(dilated_mask)
        # opacity = 0.2
        # cv2.imshow(COLORS[index], cv2.addWeighted(frame.astype(np.uint8), opacity, dilated_mask.astype(np.uint8), 1, 0))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    results = []
    for index, highlight in enumerate(highlight_masks_diluted):
        # print(COLORS[index])
        # search_mask = np.where(highlight[:, :, 2] != 255)
        # at these locations, check the hsv frame for bright pixels

        black_pixels = np.where(highlight[:, :, 2] != 255)
        cropped_frame = hsv_frame.copy()
        cropped_frame[black_pixels] = [0, 0, 0]

        min_highlight = MIN_HIGHLIGHT
        max_highlight = MAX_HIGHLIGHT
        if COLORS[index] == "black":
            min_highlight = MIN_HIGHLIGHT_BLACK
        elif COLORS[index] == "yellow":
            max_highlight = MAX_HIGHLIGHT_YELLOW
        elif COLORS[index] == "pink":
            max_highlight = MAX_HIGHLIGHT_PINK

        highlight_pixels = cv2.inRange(cropped_frame, min_highlight, max_highlight)
        high = cv2.cvtColor(highlight_pixels, cv2.COLOR_GRAY2BGR)
        # img_show(np.concatenate((cropped_frame, high), axis=0))

        highlight_coordinates = np.where(highlight_pixels > 0)

        if len(highlight_coordinates[0]) == 0:
            continue

        if COLORS[index] == "red":
            # get n biggest highlight
            groups = get_groups(highlight_pixels)
            largest_groups = groups[:15]
            # get mean per group
            # print("\n\n\nred")
            highlight_mean = []
            highlight_max = []
            new_mask = np.zeros(highlight_pixels.shape)
            for group in largest_groups:
                group_mean = np.mean(group, axis=0)
                highlight_mean.append(group_mean)
                highlight_max.append(
                    (np.max(np.array(group)[:, 0], axis=0), group_mean[1])
                )
                # print(COLORS[index], group_mean)
                for point in group:
                    new_mask[point[0], point[1]] = 255
                    highlight_pixels = new_mask

        else:
            highlight_mean = np.mean(highlight_coordinates, axis=1)
            highlight_max = [(max(highlight_coordinates[0]), highlight_mean[1])]

            # print(COLORS[index], highlight_mean)
        highlight_mean = np.array(highlight_mean)
        if (
            len(highlight_mean) > 1 and (highlight_mean == highlight_mean).all()
        ):  # check that the mean is not NaN
            if version == "max":
                results.append((COLORS[index], highlight_max, highlight_pixels))
            else:
                results.append((COLORS[index], highlight_mean, highlight_pixels))

    return results


# Convert the result
def flatten_red_balls(result):
    points = []
    colors = []
    for r in results:
        if isinstance(r[1], list):
            for point in r[1]:
                points += [point]
                colors += [r[0]]
        else:
            points += [r[1]]
            colors.append(r[0])
    return points, colors


def visualize(frame, points, colors):
    image = np.zeros((720, 1280, 3), np.uint8)
    radius = 5
    thickness = 7
    for point, color in zip(points, colors):
        image = cv2.circle(
            image,
            (int(point[1]), int(point[0])),
            radius,
            COLOR_STR_TO_BGR.get(color, (255, 255, 255)),
            thickness,
        )

    images = np.hstack((frame, image))

    cv2.imshow("side by side", images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    frame = cv2.imread("frames/masked_frame.png")

    results = detect_highlight(frame)

    # for color, coords, nani in results:
    # print(color, coords)

    # # show the highlights
    # new = frame.copy()
    # new[int(results[0][1][0]), int(results[0][1][1])] = [255, 0, 0]
    # for r in results[1][1]:
    # print(int(r[0]), int(r[1]))
    # new[int(r[0]), int(r[1])] = [255, 0, 0]
    # for r in results[2:]:
    # print(r[1][0], r[1][1])
    # new[int(r[1][0]), int(r[1][1])] = [255, 0, 0]
    # cv2.imshow("result", new)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # #
    # points, colors = flatten_red_balls(results)
    # #
    # #
    # visualize(frame, points, colors)
