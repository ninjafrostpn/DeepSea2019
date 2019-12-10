import cv2
import time

# With just reading the frames:
# 2019-12-10 09:33:40 - 2019-12-10 09:37:55 (about 5min, to overestimate)

cap = cv2.VideoCapture("IsisROV_dive148_TRANSECT.m4v")
# cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret = True
print(time.strftime("%Y-%m-%d %H:%M:%S"))
while ret:
    ret, _ = cap.read()
print(time.strftime("%Y-%m-%d %H:%M:%S"))
