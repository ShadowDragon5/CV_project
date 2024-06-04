from pathlib import Path

import cv2
from dotenv import dotenv_values

# local imports
from feature_detection import table_edges_detection
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

    out = table_edges_detection(reference_frame)
    img_show(out)

    # TODO:
    # process_video(data_path / "WSC.mp4")


def process_video(video_path: Path):
    filtered_path = video_path.with_name(
        video_path.stem + "_filtered" + video_path.suffix
    )
    reader = (
        VideoReader(filtered_path) if filtered_path.exists() else get_frames(video_path)
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


def get_frames(video_path):
    gen1 = VideoReader(video_path)
    gen2 = VideoReader(video_path)
    next(gen2)
    for f1, f2 in zip(gen1, gen2):
        yield f1 - f2


# FIXME: abstract this
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
