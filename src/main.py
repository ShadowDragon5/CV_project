from pathlib import Path

import cv2
import numpy as np
from dotenv import dotenv_values

# local imports
from feature_detection import (
    detect_baulk_line,
    get_ball_centers,
    table_edges_detection,
)
from utils import VideoReader, img_show


def main():
    env = dotenv_values()
    data_path = Path(env["DATA_DIR"])
    ref_frame_path = str(data_path / "WSC sample.png")

    print(f"Processing {ref_frame_path}...")
    reference_frame = cv2.imread(ref_frame_path, cv2.IMREAD_COLOR)

    # # scale the frame
    # scale = 0.5
    # reference_frame = cv2.resize(reference_frame, (0, 0), fx=scale, fy=scale)

    line_frame, lines = table_edges_detection(reference_frame.copy())
    mask = create_table_mask(lines, reference_frame.shape[:2])

    reference_frame = cv2.bitwise_and(reference_frame, reference_frame, mask=mask)
    centers = get_ball_centers(reference_frame)
    (baulk_rho, baulk_theta) = detect_baulk_line(reference_frame)[0]

    # Calculate the coordinates where the balls are placed on the table
    ground_points = {}
    for color, (x, _) in centers:
        if color not in ["yellow", "brown", "green"]:
            continue
        y = (baulk_rho - x * np.cos(baulk_theta)) / np.sin(baulk_theta)
        ground_points[color] = [x, int(y)]

    for _, (x, y) in centers + list(ground_points.items()):
        reference_frame[y, x] = (0, 0, 255)

    img_show(reference_frame)

    # TODO:
    # process_video(data_path / "WSC.mp4")


def create_table_mask(lines: list, shape) -> np.ndarray:
    """
    Creates a mask for the play area given the 4 lines defining the borders
    lines: list of lines in polar coordinates (rho, theta) from Hough transform
    shape: the 2D shape of the frame that the mask will be created
    """
    y, x = np.indices(shape)
    mask = np.ones(shape, dtype=np.uint8)
    for rho, theta in lines:
        # TODO: there should be a better way check which side to pick
        if rho * np.sin(theta) > 50:
            m = x * np.cos(theta) + y * np.sin(theta) > rho
        else:
            m = x * np.cos(theta) + y * np.sin(theta) < rho
        # m = (m - np.min(m)) / (np.max(m) - np.min(m)) * 255
        # m = m.astype(np.uint8)
        mask = mask * m
    return mask


def process_video(video_path: Path):
    filtered_path = video_path.with_name(
        video_path.stem + "_filtered" + video_path.suffix
    )
    # TODO: use reader to make sure the filtered file exists
    reader = (
        VideoReader(filtered_path)
        if filtered_path.exists()
        else get_filtered_frames(video_path)
    )

    # OPTIMIZE:
    for frame in VideoReader(video_path, max_count=25_000):
        cv2.imshow("Video", frame)

        key = cv2.waitKey(10)
        if (
            cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1  # window is closed
            or (key & 0xFF) == ord("q")
        ):
            cv2.destroyAllWindows()
            break


# TODO: move frame filtering logic here
def get_filtered_frames(video_path):
    gen1 = VideoReader(video_path)
    gen2 = VideoReader(video_path)
    next(gen2)
    for f1, f2 in zip(gen1, gen2):
        yield f1 - f2


# path = ".data/"
#
# # Input data
# video_in = cv2.VideoCapture(path + "WSC.mp4")
# sample_score_tag = cv2.imread(path + "WSC sample-score-tag.png")
#
# # Codec information
# fourcc_in = int(video_in.get(cv2.CAP_PROP_FOURCC))
# codec_string_in = fourcc_in.to_bytes(4, byteorder=sys.byteorder).decode().upper()
# # print(codec_string_in)
#
# # General interest
# fps_in = video_in.get(cv2.CAP_PROP_FPS)
# modulus = 2**2  # cycle of frames skipped
# fps_out = fps_in  # fps_in/modulus #rescaled fps, can be float
# downscale_ratio = 1 / 2**3  # downscale in resolution
# width_out = int(video_in.get(3) * downscale_ratio)  # rounding with int or round?
# height_out = int(video_in.get(4) * downscale_ratio)
#
# # Output video data
# fourcc_out = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # works on Nico's Linux
# # fourcc_out = cv2.VideoWriter_fourcc('F','M','P','4')
# # fourcc_out = cv2.VideoWriter_fourcc('X','2','6','4')
# file_extension_out = "mp4"  # try avi
# file_name_out = "Course-Project-out"
# path_out = path + file_name_out + "." + file_extension_out
# video_out = cv2.VideoWriter(path_out, fourcc_out, fps_in, (width_out, height_out))
#
# # Iterating flags and counter
# process_only_first_frame = False
# first_frame_flag = True
# frame_counter = 1
# frame_counter_mod = 0  # counter to drop frames by modulus
# relevant_frame_counter = (
#     0  # counter of frame that passed the filter (workload approximation)
# )
# frame_out = np.zeros((height_out, width_out, 3), np.uint8)
#
# ret, frame_in = video_in.read()
# # Main loop
# while ret and frame_counter < 25000:  # around minute 12
#     if process_only_first_frame and not first_frame_flag:
#         break
#
#     compute_flag = frame_counter_mod == 0  # Condition to execute main on frame
#     if compute_flag:
#         # Downsize for faster execution
#         frame = cv2.resize(frame_in, (width_out, height_out))
#
#         # Decide if drop frame
#         frame_score_tag = extract_score_tag(frame)
#         frame_is_relevant = compare_score_tag(sample_score_tag, frame_score_tag)
#         # print(frame_is_relevant)
#         write_flag = frame_is_relevant  # drop frame from writing if irrelevant
#         relevant_frame_counter += frame_is_relevant
#
#         # Call to important part of the code and update frame_out
#         frame_out = main_stub(frame)
#
#         # Visualize process
#         # cv2.imshow('output_frame',output_frame)
#
#     # Out
#     if first_frame_flag and process_only_first_frame:
#         cv2.imwrite(path + "first_frame.png", frame_out)
#     elif (
#         write_flag and frame_counter_mod == 0
#     ):  # Drop writing repeat for modulus frames
#         video_out.write(
#             frame_out
#         )  # for every frame in there is a frame out (realtime), or drop mod frames
#
#     # Update next frame conditions
#     first_frame_flag = False
#     previous_frame = frame_out
#     frame_counter += 1
#     frame_counter_mod = (frame_counter_mod + 1) % modulus
#     ret, frame_in = video_in.read()
#
# print(relevant_frame_counter)
#
# # Close file
# video_in.release()
# video_out.release()

if __name__ == "__main__":
    main()
