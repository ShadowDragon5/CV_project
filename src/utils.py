from pathlib import Path

import cv2
from cv2.typing import MatLike


def img_show(img, title="Title") -> None:
    cv2.imshow(title, img)

    while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
        key = cv2.waitKey(100)

        # q key closes the window
        if (key & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break


class VideoReader:
    def __init__(self, video_path: str | Path, max_count=-1) -> None:
        self.count = max_count
        self.video = cv2.VideoCapture(str(video_path))

        if not self.video.isOpened():
            print("Error openning video file")

    def __iter__(self):
        return self

    def __next__(self) -> MatLike:
        _, frame = self.video.read()

        if frame is None or self.count == 0:
            self.video.release()
            raise StopIteration

        self.count -= 1
        return frame
