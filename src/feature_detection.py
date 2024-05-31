import numpy as np

from hough_line import hough_lines


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

    lines = hough_lines(green_brown_edges, num_lines=4)

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


# %%
# Function that detects the lines markings on the table
def baulk_line_detection(frame_BGR):
	
	width, height, _ = frame_BGR.shape

    frame_HSV = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HSV)
    	
    # Filter green color
    green_HSV_low = np.array([50, 200, 110])  # Hardcoded green HSV value
    green_HSV_high = np.array([70, 255, 210])
    frame_green = cv2.inRange(frame_HSV, green_HSV_low, green_HSV_high)

	frame_HSL = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HLS)

    # Filter white color
    white_HSL_low = np.array([0, 255, 0])  # Hardcoded white HSL value
    brown_HSL_high = np.array([255, 255, 255])
    frame_white = cv2.inRange(frame_HSL, white_HSL_low, brown_HSL_high)

    # Noise filtering
    # frame_green = cv2.morphologyEx(frame_green, cv2.MORPH_OPEN, kernel)
    # frame_green = cv2.morphologyEx(frame_green, cv2.MORPH_CLOSE, kernel)

    # Edge detection between green and brown
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    green_dilate = cv2.dilate(frame_green, kernel)
    #white_dilate = cv2.dilate(frame_white, kernel)
    green_white_edges = cv2.bitwise_and(green_dilate, white_dilate)

    # Edge detection using morphological gradient
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    # frame_brown = cv2.morphologyEx(frame_brown, cv2.MORPH_GRADIENT, kernel)

    baulk_line = hough_lines(green_brown_edges, num_lines=1)
	
	return baulk_line
