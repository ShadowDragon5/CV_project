import cv2
import numpy as np

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
