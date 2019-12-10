import cv2
import numpy as np
import pandas as pd
import time

# With just reading the frames:
#   for all: 2019-12-10 09:33:40 - 2019-12-10 09:37:55 (about 5 min, to overestimate)
# With averaging frame colours:
#   for 1000: 2019-12-10 09:59:44 - 2019-12-10 10:00:03 (maybe 30 sec)
#   for 1000, taking every second pixel: 2019-12-10 10:03:20 - 2019-12-10 10:03:26 (about 7 sec -> ~11 min for all?)
#   for all, taking every second pixel: 2019-12-10 10:13:58 - 2019-12-10 10:25:18 (Almost 12 min)

cap = cv2.VideoCapture("IsisROV_dive148_TRANSECT.m4v")
capn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(capn)
# cap.set(cv2.CAP_PROP_POS_FRAMES, capn - 1000)
ret = True
avgcol = -np.ones([capn, 3], dtype=np.float64)
print(time.strftime("%Y-%m-%d %H:%M:%S"))
i = 0
while ret:
    ret, frame = cap.read()
    if ret:
        avgcol[i] = np.mean(frame[::2, ::2], axis=(0, 1), dtype=np.float64)
        i += 1
print(time.strftime("%Y-%m-%d %H:%M:%S"))
dfout = pd.DataFrame({"Frame": np.arange(len(avgcol)),
                      **{"Avg" + colname: avgcol[:, i] for i, colname in enumerate(["R", "G", "B"])}})
dfout.to_csv("AvgCol.csv", index=False, columns=["Frame", "AvgR", "AvgG", "AvgB"])
print(time.strftime("%Y-%m-%d %H:%M:%S"))
