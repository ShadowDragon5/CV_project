from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from DLT import DLT
from dotenv import dotenv_values
from feature_detection import (
    detect_baulk_line,
    get_ball_centers,
    table_edges_detection,
)
from highlight_detection import detect_highlight
from light_related_stuff import compute_ball_center_from_specular_reflection
from preprocessing import compare_score_tag, extract_score_tag
from utils import VideoReader, img_show


def main():
    env = dotenv_values()
    data_path = Path(env["DATA_DIR"])
    ref_frame_path = str(data_path / "WSC sample.png")

    print(f"Processing {ref_frame_path}...")
    reference_frame = cv2.imread(ref_frame_path, cv2.IMREAD_COLOR)

    # Find table boundaries and create a mask to hide irrelevant frame data
    corners = table_edges_detection(reference_frame.copy())
    mask = cv2.fillPoly(np.zeros(reference_frame.shape[:2], dtype=np.uint8), corners, 1)

    # Show the selected area
    mask_neg = cv2.fillPoly(
        np.ones(reference_frame.shape[:2], dtype=np.uint8), corners, 0
    )
    reference_frame_neg = cv2.bitwise_and(
        reference_frame, reference_frame, mask=mask_neg
    )
    # img_show(reference_frame_neg)

    # apply the mask
    masked_frame = cv2.bitwise_and(reference_frame, reference_frame, mask=mask)
    # img_show(masked_frame)

    # Find reference points (ball centers and baulk line)
    centers = get_ball_centers(masked_frame)
    (baulk_rho, baulk_theta) = detect_baulk_line(masked_frame)

    # Calculate the coordinates where the balls are placed on the table
    ground_points = {}
    for color, (x, _) in centers.items():
        if color not in ["yellow", "brown", "green"]:
            continue

        # $y = \frac{\rho - x\cos\theta}{\sin\theta}$
        y = (baulk_rho - x * np.cos(baulk_theta)) / np.sin(baulk_theta)
        ground_points[color + "_table"] = [x, int(y)]

    image_points = centers | ground_points

    x_points = np.array(
        [
            image_points["yellow"],
            image_points["yellow_table"],
            image_points["brown"],
            image_points["brown_table"],
            image_points["green"],
            image_points["green_table"],
            image_points["blue"],
            # play area corners
            *corners[0],
        ]
    )

    # http://www.fcsnooker.co.uk/billiards/the_table_and%20table_markings.htm
    # we are detecting the play area +2 inches for the cushions
    X_points = np.array(
        [
            # balls
            [-0.292, 1.0475, 0.02625],  # yellow
            [-0.292, 1.0475, 0.0],  # yellow_table
            [0.0, 1.0475, 0.02625],  # brown
            [0.0, 1.0475, 0.0],  # brown_table
            [0.292, 1.0475, 0.02625],  # green
            [0.292, 1.0475, 0.0],  # green_table
            [0.0, 0.0, 0.02625],  # blue
            # play area + cushions corners
            [-0.934, 1.829, 0.03],  # top-left
            [0.934, 1.829, 0.03],  # top-right
            [0.934, -1.829, 0.03],  # bottom-right
            [-0.934, -1.829, 0.03],  # bottom-left
        ]
    )

    # DLT
    P = DLT(X_points, x_points)
    M = P[:, :3]  # Rotation matrix of the camera
    # P_inv = np.linalg.pinv(P)
    camera = -np.linalg.inv(M).dot(P[:, 3].transpose())

    LIGHT_HARDCODED = [0, -3, 6]

    # for _, (x, y) in image_points.items():
    #     reference_frame[y, x] = (0, 0, 255)
    # img_show(reference_frame, "with centers")

    # Video processing
    video_path = data_path / "WSC.mp4"

    filtered_path = video_path.with_name(
        video_path.stem + "_filtered" + video_path.suffix
    )

    # Create a filtered video if it doesn't exist (might take ~15mins)
    if not filtered_path.exists():
        sample_score_tag = extract_score_tag(reference_frame)
        filter_video(video_path, filtered_path, sample_score_tag)

    for frame in VideoReader(filtered_path, max_count=25_000):
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # specular_out = detect_highlight(masked_frame)
        # res = compute_ball_center_from_specular_reflection(
        #     P, camera, LIGHT_HARDCODED, specular_out
        # )

        # for _, (x, y, _) in res:
        #     reference_frame[y, x] = (0, 0, 255)

        # display result
        cv2.imshow("Video", masked_frame)

        # window closing
        key = cv2.waitKey(10)
        if (
            cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1  # window is closed
            or (key & 0xFF) == ord("q")
        ):
            cv2.destroyAllWindows()
            break


def filter_video(in_path: Path, out_path: Path, sample_tag: MatLike):
    reader = VideoReader(in_path)

    fps = reader.video.get(cv2.CAP_PROP_FPS)
    shape = (
        int(reader.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(reader.video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, shape)

    # NOTE: scaling down the frame is slower (40s vs 36s for first 13mins)
    for frame in reader:
        frame_score_tag = extract_score_tag(frame)
        if compare_score_tag(sample_tag, frame_score_tag, 0.05):
            writer.write(frame)

    writer.release()


if __name__ == "__main__":
    main()
