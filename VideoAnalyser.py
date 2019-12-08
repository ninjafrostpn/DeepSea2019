import cv2
from glob import glob
import numpy as np
import pandas as pd
import pygame
from pygame.locals import *
from scipy.ndimage.filters import convolve
import time

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
MAGENTA = np.int32([255, 0, 255])

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
pos, frame = getnewframe(0)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
skipspeed = 5

# Create DataFrame for per-frame output data
# The ScaleOK column ought to be switched to False where the script has picked up incorrectly
# The Faunal columns give counts of each individual once, hopefully on the first frame they were spotted
scalestats = ["ScalePX", "ScalePX/m", "UScaleX", "UScaleY", "LScaleX", "LScaleY"]
faunanames = ["Shreemp", "Extendoworm", "Clamaybe", "Anemone", "Ophi", "Feesh", "Squeed"]
# For ScaleOK, -1 represents unchecked, -2 or 2 represent bad/good autocheck, 0 or 1 represent confirmed bad/good
coldefaults = {"Frame": np.arange(0, capn, skipspeed), **{col: np.nan for col in dfenv.columns},
               "ScaleOK": -1, **{i: np.nan for i in scalestats},
               "LastEdited": "nan", **{name: 0 for name in faunanames}}
try:
    # TODO: Find a sort that actually works with numbered files with 2 digits or more
    csvnames = sorted(glob("AllData{:.0f}-*.csv".format(skipspeed)))
    if len(csvnames) == 0:
        raise FileNotFoundError
    csvinname = csvnames[-1]
    dfout = pd.read_csv(csvinname)
    csvoutname = "AllData{:.0f}-{}.csv".format(skipspeed,
                                               int(csvinname[csvinname.index("-") + 1:csvinname.index(".")]) + 1)
    print("{} frames displayed, {} in".format(len(dfout["Frame"]), np.ceil(capn / skipspeed)), csvoutname)
    if len(dfout["Frame"]) != np.ceil(capn / skipspeed):
        raise Exception("{}'s Frame Count or Resolution Doesn't Match Analyser Setting".format(csvoutname))
    coldefaultskeys = np.object_(list(coldefaults.keys()))
    colsinmask = np.isin(coldefaultskeys, dfout.columns)
    if not np.all(colsinmask):
        print(coldefaultskeys[~colsinmask], "not found, adding")
        dfout = dfout.assign(**{key: coldefaults[key] for key in coldefaultskeys[~colsinmask]})
    print("Loaded in", csvinname)
except FileNotFoundError:
    csvoutname = "AllData{:.0f}-0.csv".format(skipspeed)
    print("{} frames displayed and in".format(np.ceil(capn / skipspeed)), csvoutname)
    print("No {} found, creating new DataFrame".format(csvoutname))
    dfout = pd.DataFrame(coldefaults)
dataoutindex = np.argwhere(dfout["Frame"] == pos)[0][0]

try:
    while True:
        if K_SPACE in keys and K_SPACE not in prevkeys:
            # Spacebar to toggle pause
            pause = not pause
            # Line up playback position with multiple of skipspeed if skipping has occurred while paused
            if pos == cap.get(cv2.CAP_PROP_POS_FRAMES):
                pos, frame = getnewframe()
        if K_TAB in keys and K_TAB not in prevkeys:
            # Tab to toggle video reticle
            showreticle = not showreticle
        if K_RSHIFT in keys and K_RSHIFT not in prevkeys:
            # Right shift to toggle decision on Scale OKness
            oldverdict = dfout.loc[dataoutindex, "ScaleOK"]
            if oldverdict <= 0:
                dfout.loc[dataoutindex, "ScaleOK"] = 1
            else:
                dfout.loc[dataoutindex, "ScaleOK"] = 0
            dfout.loc[dataoutindex, "LastEdited"] = time.strftime("%Y-%m-%d %H:%M:%S")
            print("ScaleOK:", dfout.loc[dataoutindex, "ScaleOK"] > 0)
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
        # Get the row to write the output data to
        dataoutindex = np.argwhere(dfout["Frame"] == pos)[0][0]
        # Clear the window
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
            # If you can see the dots, it gives its opinion on their OKness according to x difference
            # (if you've not offered an opinion before, but will change its own verdicts)
            oldverdict = dfout.loc[dataoutindex, "ScaleOK"]
            if np.isnan(oldverdict) or abs(oldverdict) == 2:
                # Offers good (2) if x difference is less than y difference over 2, else bad (-2)
                # TODO: Add check for one dot being deflected up or down
                dfout.loc[dataoutindex, "ScaleOK"] = [-2, 2][int(abs(np.divide(*(upperdot - lowerdot))) < 0.5)]
        # Store the scalebar stats
        dfout.loc[dataoutindex, scalestats] = [scalebar, scalebar * 10, *upperdot, *lowerdot]
        dfout.loc[dataoutindex, "LastEdited"] = time.strftime("%Y-%m-%d %H:%M:%S")
        # Show Video stats on screen
        # TODO: Add little loading-bar/range-bar -esque elements to accompany data
        writedataline("~ VIDEO ~", 1)
        writedataline("Frame: {:05.0f}/{:.0f}".format(pos, capn), 2)
        writedataline("Time: {:04.0f}s/{:.0f}s".format(pos / capfps, capn / capfps), 3)
        frametime = pos / capfps
        if int(frametime) in dfenv["VideoTimeSecs"]:
            dataenvindex = np.argwhere(int(frametime) == dfenv["VideoTimeSecs"])[0][0]
            dataenvtime = int(frametime)
        else:
            dataenvindex = np.searchsorted(dfenv["VideoTimeSecs"], int(frametime)) - 1
            dataenvtime = dfenv["VideoTimeSecs"][dataenvindex]
        # Store the environmental stats from the xlsx in the output csv
        dfout.loc[dataoutindex, dfenv.columns] = dfenv.loc[dataenvindex]
        dfout.loc[dataoutindex, "LastEdited"] = time.strftime("%Y-%m-%d %H:%M:%S")
        writedataline("~ ENVDATA ~", 5, GREEN)
        writedataline("Time: {:04.0f}s/{:.0f}s".format(dataenvtime, max(dfenv["VideoTimeSecs"])), 6, GREEN)
        writedataline("Dist: {: 04.0f}.{:02.0f}m".format(dfenv["Dist"][dataenvindex],
                                                         abs(dfenv["Dist"][dataenvindex] * 100) % 100),
                      7, GREEN)
        writedataline("Depth: {:.2f}m".format(dfenv["Depth"][dataenvindex]), 8, GREEN)
        writedataline("Temp: {:.4f}°C".format(dfenv["Temp"][dataenvindex]), 9, GREEN)
        writedataline("~ CALC ~", 11, CYAN)
        writedataline("px per m: {:.3f}".format(scalebar * 10), 12, CYAN)
        writedataline("~ DATAOUT ~", 14, MAGENTA)
        verdict = dfout.loc[dataoutindex, "ScaleOK"]
        writedataline("ScaleOK: {}{}".format(verdict > 0, " (?)" if not (verdict in [0, 1]) else ""), 15, MAGENTA)
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
    unsaved = True
    while unsaved:
        try:
            print("Saving DataFrame to", csvoutname)
            dfout.to_csv(csvoutname, na_rep="nan", index=False,
                         columns=["Frame", "VideoTimeSecs", "VideoTime", "ActualTime", "Dist", "Lat", "Lon",
                                  *faunanames, "Depth", "Temp", "Salinity", *scalestats, "ScaleOK", "LastEdited"])
            unsaved = False
        except PermissionError:
            print(csvoutname, "might be open elsewhere, waiting for you to close it...")
            time.sleep(5)
    print("Save completed")

"""
https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
"""
