import PIL.Image
import cv2
import glob
import os
from datetime import datetime

import numpy as np
import torchvision.transforms

VIDEO_PATH = r"D:\RawData\014\background_2\profile\normal\right_60\VID_20230305_154038.mp4"
SAVE_PATH = r'D:\RawData\014\background_2\profile\normal\right_60'


def video_to_frames(path):
    videoCapture = cv2.VideoCapture()
    videoCapture.open(path)
    # 帧率
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # 总帧数
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", int(fps), "frames=", int(frames))

    for i in range(int(frames)):
        print(i + 1)
        ret, frame = videoCapture.read()
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.RandomVerticalFlip(p=1),
             torchvision.transforms.RandomHorizontalFlip(p=1)]
        )
        frame = np.asarray(transform(PIL.Image.fromarray(frame)))
        cv2.imwrite("%s/%d.jpg" % (SAVE_PATH, i + 1), frame)
    return


if __name__ == '__main__':
    t1 = datetime.now()
    video_to_frames(VIDEO_PATH)
    t2 = datetime.now()
    print("Time cost = ", (t2 - t1))
    print("SUCCEED !!!")
