import cv2
import glob
import os
from datetime import datetime


def video_to_frames(path):
    videoCapture = cv2.VideoCapture()
    videoCapture.open(path)
    # 帧率
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # 总帧数
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", int(fps), "frames=", int(frames))

    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        cv2.imwrite("frames/%d.jpg" % (i + 1), frame)
    return


if __name__ == '__main__':
    t1 = datetime.now()
    video_to_frames("1.mp4")
    t2 = datetime.now()
    print("Time cost = ", (t2 - t1))
    print("SUCCEED !!!")
