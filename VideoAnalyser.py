import cv2
import numpy as np
import pandas as pd
import pygame
from pygame.locals import *
from scipy.ndimage.filters import convolve

# Function for converting HH:MM:SS to total seconds
timetosec = np.vectorize(lambda x: (((x.hour * 60) + x.minute) * 60) + x.second)

# Read data in from the xlsx file as a pandas DataFrame (remove metadata in first 6 rows)
dfenv = pd.read_excel("SOES6008_coursework_DATA.xlsx", skiprows=6)
dfenv.columns = ["ActualTime", "VideoTime", "Lat", "Lon", "Dist", "Depth", "Temp", "Salinity"]
dfenv = dfenv.assign(VideoTimeSecs=timetosec(dfenv["VideoTime"]))

# Initialise window
pygame.init()

# Keys pressed in the frame just shown...
keys = set()
# ...and the one before that (for good edge detection)
prevkeys = set()

# Colours
WHITE = np.int32([255] * 3)
GREEN = np.int32([0, 255, 0])
CYAN = np.int32([0, 255, 255])

# Font for drawing text with
textfont = pygame.font.Font(None, 30)

# Initialise all window and video-getting tools
cap = cv2.VideoCapture("IsisROV_dive148_TRANSECT.m4v")
capn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
capfps = cap.get(cv2.CAP_PROP_FPS)
capw, caph = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
capsize = np.int32([capw, caph])
screensize = np.int32([capw + 300, caph])
screen = pygame.display.set_mode(screensize)
viewport = screen.subsurface([0, 0, capw, caph])
dataport = screen.subsurface([capw, 0, 300, caph])
pygame.display.set_caption("VIDEO", "Video")
print("{:.0f} frames ({:.0f} x {:.0f}) at {} FPS".format(capn, capw, caph, capfps))


def getnewframe(pos=None):
    if pos is not None:
        if pos < 0:
            pos = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ret, frame = cap.read()
    if not ret:
        print("Oh No")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(np.fliplr(frame))
    return pos, frame


def writedataline(text, lineno, col=WHITE):
    dataport.blit(textfont.render(text, True, col), (20, 30 * lineno))


# Blob-shaped filter used to find red blobs
redblobfinder = - 5 * np.ones([17, 17])
redblobfinder = cv2.circle(redblobfinder, (8, 8), 6, 10, -1)

# Set initial state of playback
pause = True
showreticle = True
pos, frame = getnewframe(9200)
cap.set(cv2.CAP_PROP_POS_FRAMES, 9200)
skipspeed = 5

# Create DataFrame for per-frame output data
# The ScaleOK column ought to be switched to False where the script has picked up incorrectly
# The Faunal columns give counts of each individual once, hopefully on the first frame they were spotted
faunanames = ["Shreemp", "Extendoworm", "Clamaybe", "Anemone", "Ophi", "Feesh", "Squeed"]
coldefaults = {"Frame": np.arange(0, capn, skipspeed), **{col: np.nan for col in dfenv.columns},
               "ScalePx": np.nan, "ScaleOK": True, "LastEdited": "nan", **{name: 0 for name in faunanames}}
try:
    dfout = pd.read_csv("AllData.csv")
    print("{} frames displayed, {} in AllData.csv".format(len(dfout["Frame"]), np.ceil(capn / skipspeed)))
    if len(dfout["Frame"]) != np.ceil(capn / skipspeed):
        raise Exception("AllData.csv's Frame Count or Resolution Doesn't Match Analyser Setting")
    coldefaultskeys = np.object_(list(coldefaults.keys()))
    colsinmask = np.isin(coldefaultskeys, dfout.columns)
    if not np.all(colsinmask):
        print(coldefaultskeys[~colsinmask], "not found, adding")
        dfout = dfout.assign(**{key: coldefaults[key] for key in coldefaultskeys[~colsinmask]})
    print("Loaded in AllData.csv")
except FileNotFoundError:
    print("{} frames displayed and in AllData.csv".format(np.ceil(capn / skipspeed)))
    print("No AllData.csv found, creating new DataFrame")
    dfout = pd.DataFrame(coldefaults)

try:
    while True:
        if K_SPACE in keys and K_SPACE not in prevkeys:
            # Spacebar to toggle pause
            pause = not pause
        if K_TAB in keys and K_TAB not in prevkeys:
            # Tab to toggle video reticle
            showreticle = not showreticle
        if K_RIGHT in keys:
            # Skip forward with right arrow key
            if pause:
                # Skip is slower when paused
                pos += skipspeed
            else:
                # 10x faster when unpaused
                pos += 10 * skipspeed
            pos, frame = getnewframe(pos)
        elif K_LEFT in keys:
            # Skip backward with left arrow key
            if pause:
                pos -= skipspeed
            else:
                pos -= 10 * skipspeed
            pos, frame = getnewframe(pos)
        elif not pause:
            # Play forward by getting a certain number of frames according to skipspeed, and only showing the last one
            # (elif, not if, to allow skipping back to frame 0)
            for i in range(skipspeed):
                pos, frame = getnewframe()
        screen.fill(0)
        pframe = pygame.surfarray.make_surface(frame)
        # Look for the red dots in the "middle half" of the image area
        redness = np.sum(frame[330:700, 144:432] * [1, -0.5, -0.5], axis=2)
        dotlikeness = convolve(redness, redblobfinder, mode="constant", cval=0)
        # First-guess upper and lower dots' positions
        # (Includes likelihood of the two dots being in the same x location)
        upperdot = np.int32(np.unravel_index(np.argmax(dotlikeness[:, :144] + dotlikeness[:, :143:-1] / 5),
                                             (700, 144))) + (330, 144)
        lowerdot = np.int32(np.unravel_index(np.argmax(dotlikeness[:, 144:] + dotlikeness[:, 143::-1] / 5),
                                             (700, 144))) + (330, 288)
        # Draw video frame to screen
        viewport.blit(pframe, (0, 0))
        scalebar = np.linalg.norm(upperdot - lowerdot)
        if showreticle:
            # Draw dot-finder reticle
            pygame.draw.rect(viewport, WHITE, [330, 144, 370, 288], 1)
            pygame.draw.line(viewport, WHITE, [330, 288], [700, 288], 1)
            pygame.draw.circle(viewport, WHITE, upperdot, 20, 1)
            pygame.draw.circle(viewport, WHITE, lowerdot, 20, 1)
            pygame.draw.line(viewport, WHITE, lowerdot, upperdot, 1)
            pygame.draw.rect(viewport, WHITE, [20, 20, scalebar, scalebar], 1)
            viewport.blit(textfont.render("10 x 10", True, WHITE), [20, 20 + scalebar])
        # TODO: Add little loading-bar/range-bar -esque elements to accompany data
        writedataline("~ VIDEO ~", 1)
        writedataline("Frame: {:05.0f}/{:.0f}".format(pos, capn), 2)
        writedataline("Time: {:04.0f}s/{:.0f}s".format(pos / capfps, capn / capfps), 3)
        frametime = pos / capfps
        if int(frametime) in dfenv["VideoTimeSecs"]:
            dataindex = np.argwhere(int(frametime) == dfenv["VideoTimeSecs"])[0][0]
            datatime = int(frametime)
        else:
            dataindex = np.searchsorted(dfenv["VideoTimeSecs"], int(frametime)) - 1
            datatime = dfenv["VideoTimeSecs"][dataindex]
        writedataline("~ DATA ~", 5, GREEN)
        writedataline("Time: {:04.0f}s/{:.0f}s".format(datatime, max(dfenv["VideoTimeSecs"])), 6, GREEN)
        writedataline("Dist: {: 04.0f}.{:02.0f}m".format(dfenv["Dist"][dataindex], abs(dfenv["Dist"][dataindex] * 100) % 100),
                      7, GREEN)
        writedataline("Depth: {:.2f}m".format(dfenv["Depth"][dataindex]), 8, GREEN)
        writedataline("Temp: {:.4f}°C".format(dfenv["Temp"][dataindex]), 9, GREEN)
        writedataline("~ CALC ~", 11, CYAN)
        writedataline("px per cm: {:.3f}".format(scalebar / 10), 12, CYAN)
        pygame.display.flip()
        prevkeys = keys.copy()
        for e in pygame.event.get():
            if e.type == QUIT:
                quit()
            elif e.type == KEYDOWN:
                keys.add(e.key)
                if e.key == K_ESCAPE:
                    quit()
            elif e.type == KEYUP:
                keys.discard(e.key)
finally:
    print("Saving DataFrame to AllData.csv")
    dfout.to_csv("AllData.csv", na_rep="nan", index=False,
                 columns=["Frame", "VideoTimeSecs", "VideoTime", "ActualTime", "Dist", "Lat", "Lon", *faunanames,
                          "Depth", "Temp", "Salinity", "ScalePx", "ScaleOK", "LastEdited"])

"""
https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
"""
