import cv2
from glob import glob
import numpy as np
import pandas as pd
import pygame
from pygame.locals import *
import re
from scipy.ndimage.filters import convolve
import time

# Initialise window
pygame.init()

# Function for converting HH:MM:SS to total seconds
timetosec = np.vectorize(lambda x: (((x.hour * 60) + x.minute) * 60) + x.second)

# Extracts version number from a filename
filenomatcher = re.compile(r"-[0-9]*\.")
getfileno = lambda x: int(filenomatcher.search(x)[0][1:-1])

# Keys pressed in the frame just shown...
keys = set()
# ...and the one before that (for good edge detection)
prevkeys = set()
numberkeys = [i for i in range(K_1, K_9 + 1)] + [K_0]

# Colours
WHITE = np.int32([255] * 3)
GREEN = np.int32([0, 255, 0])
CYAN = np.int32([0, 255, 255])
MAGENTA = np.int32([255, 0, 255])
RED = np.int32([255, 0, 0])
BLUE = np.int32([0, 0, 255])

# Font for drawing text with
textfont = pygame.font.Font(None, 30)

# Initialise all window and video-getting tools
cap = cv2.VideoCapture("IsisROV_dive148_TRANSECT.m4v")
capn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
capfps = cap.get(cv2.CAP_PROP_FPS)
capw, caph = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
capsize = np.int32([capw, caph])
screensize = np.int32([capw + 550, caph])
screen = pygame.display.set_mode(screensize)
dataport = screen.subsurface([0, 0, 300, caph])
viewport = screen.subsurface([300, 0, capw, caph])
faunaport = screen.subsurface([capw + 300, 0, 250, caph])
pygame.display.set_caption("VIDEO", "Video")
print("{:.0f} frames ({:.0f} x {:.0f}) at {} FPS".format(capn, capw, caph, capfps))


# Read the next frame from the video, moving to location specified by pos if required
def getnewframe(newpos=None):
    if newpos is not None:
        newpos = min(max(0, newpos), (capn // skipspeed) * skipspeed)
        cap.set(cv2.CAP_PROP_POS_FRAMES, newpos)
    newpos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    newret, newframe = cap.read()
    if not newret:
        print("Oh No")
    newframe = cv2.cvtColor(newframe, cv2.COLOR_BGR2RGB)
    newframe = np.rot90(np.fliplr(newframe))
    return newpos, newframe


# Write to the data display panel
def writedataline(text, lineno, col=WHITE, port=dataport):
    port.blit(textfont.render(text, True, col), (20, 30 * lineno))


# Blob-shaped filter used to find red blobs
redblobfinder = - 5 * np.ones([17, 17])
redblobfinder = cv2.circle(redblobfinder, (8, 8), 6, 10, -1)

# Set initial state of interface
pause = True
showreticle = True
startpos = 0
skipspeed = 10000
pos, frame = getnewframe(startpos)
cap.set(cv2.CAP_PROP_POS_FRAMES, startpos)
faunaindex = 0
frameoffset = startpos % skipspeed

# Read data in from the xlsx file as a pandas DataFrame (remove metadata in first 6 rows)
dfenv = pd.read_excel("SOES6008_coursework_DATA.xlsx", skiprows=6)
dfenv.columns = ["ActualTime", "VideoTime", "Lat", "Lon", "Dist", "Depth", "Temp", "Salinity"]
dfenv = dfenv.assign(VideoTimeSecs=timetosec(dfenv["VideoTime"]))
minigraph = pygame.Surface([300, caph - 420])
scaledsalinity = 30 - (dfenv["Salinity"] - min(dfenv["Salinity"])) * 100
scaledtemp = 126 - (dfenv["Temp"] - min(dfenv["Temp"])) * 300
scaleddepth = (dfenv["Depth"] - min(dfenv["Depth"])) * 1.2
for i in range(dfenv.shape[0] - 1):
    pygame.draw.line(minigraph, GREEN / 2,
                     [dfenv["Dist"][i], scaledsalinity[i]],
                     [dfenv["Dist"][i + 1], scaledsalinity[i + 1]])
for i in range(dfenv.shape[0] - 1):
    pygame.draw.line(minigraph, RED,
                     [dfenv["Dist"][i], scaledtemp[i]],
                     [dfenv["Dist"][i + 1], scaledtemp[i + 1]])
for i in range(dfenv.shape[0] - 1):
    pygame.draw.line(minigraph, BLUE,
                     [dfenv["Dist"][i], scaleddepth[i]],
                     [dfenv["Dist"][i + 1], scaleddepth[i + 1]])

# Create DataFrame for per-frame output data
# The ScaleOK column ought to be switched to False where the script has picked up incorrectly
# The Faunal columns give counts of each individual once, hopefully on the first frame they were spotted
scalestats = ["ScalePX", "ScalePX/m", "AreaEstimate", "UScaleX", "UScaleY", "LScaleX", "LScaleY"]
faunanames = ["Shreemp", "Extendoworm", "Clamaybe", "Anemone", "Ophi", "Feesh", "Squeed"]
# For ScaleOK, -1 represents unchecked, -2 or 2 represent bad/good autocheck, 0 or 1 represent confirmed bad/good
coldefaults = {"Frame": np.arange(0, capn, skipspeed), **{col: np.nan for col in dfenv.columns},
               "ScaleOK": -1, **{i: np.nan for i in scalestats},
               "LastEdited": "nan", **{name: 0 for name in faunanames}}
try:
    # Filenames are of format "AllDataX[Y]-Z.csv", where X is the skipspeed, Y is the frameoffset, Z is the version nr
    print(skipspeed, frameoffset, r"AllData{:.0f}\[{}\]-*.csv".format(skipspeed, frameoffset))
    csvnames = sorted(glob("AllData{:.0f}({})-*.csv".format(skipspeed, frameoffset)), key=getfileno)
    if len(csvnames) == 0:
        csvinname = "Matching Files"
        raise FileNotFoundError
    csvinname = csvnames[-1]
    dfout = pd.read_csv(csvinname)
    csvoutname = "AllData{:.0f}({})-{}.csv".format(skipspeed, frameoffset, getfileno(csvinname) + 1)
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
    csvoutname = "AllData{:.0f}({})-0.csv".format(skipspeed, frameoffset)
    print("No {} found, creating new DataFrame".format(csvinname))
    print("{} frames displayed and in".format(np.ceil(capn / skipspeed)), csvoutname)
    dfout = pd.DataFrame(coldefaults)
dataoutindex = np.argwhere(dfout["Frame"] == pos)[0][0]

# TODO: (After assignment) add help interface, automatic graph scaling, etc
try:
    while True:
        mousepos = np.int32(pygame.mouse.get_pos())
        # TODO: Actually add facility to record animal sightings, since that's the point of this application!
        numberspressed = list(keys.intersection(numberkeys))
        if len(numberspressed) > 0:
            # Number keys select that numbered faunal record
            faunaindex = numberkeys.index(numberspressed[0])
        # These two are placed here (before updat of dataoutindex)so that they get the previous frame
        # (when the key was pressed) before the next one is drawn
        if K_UP in keys and K_UP not in prevkeys:
            # Up key adds a count for the selected faunal record
            dfout.loc[dataoutindex, faunanames[faunaindex]] += 1
            dfout.loc[dataoutindex, "LastEdited"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if K_DOWN in keys and K_DOWN not in prevkeys:
            # Up key removes a count for the selected faunal record
            dfout.loc[dataoutindex, faunanames[faunaindex]] -= 1
            dfout.loc[dataoutindex, "LastEdited"] = time.strftime("%Y-%m-%d %H:%M:%S")
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
            # Differing methods for skipping used at different skipspeed ranges due to differences in retrieval time
            if skipspeed <= 200:
                for i in range(skipspeed):
                    pos, frame = getnewframe()
            else:
                pos, frame = getnewframe(pos + skipspeed)
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
            pygame.draw.circle(viewport, MAGENTA, upperdot, 20, 1)
            pygame.draw.circle(viewport, MAGENTA, lowerdot, 20, 1)
            pygame.draw.line(viewport, CYAN, lowerdot, upperdot, 1)
            viewportmousepos = mousepos - viewport.get_offset()
            if viewport.get_rect().collidepoint(viewportmousepos):
                pygame.draw.rect(viewport, WHITE, [*viewportmousepos, scalebar, scalebar], 1)
                viewport.blit(textfont.render("10 x 10", True, WHITE), viewportmousepos + (0, scalebar))
            else:
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
        # (L[px] / 10[cm]) * (100[cm] / [m]) = 10L[px] / [m]
        # (X[px] * Y [px]) / ((10L[px] / [m]) ** 2) = 0.01(XY / (L**2))[m**2]
        dfout.loc[dataoutindex, scalestats] = [scalebar, 10 * scalebar, (0.01 * capw * caph) / (scalebar ** 2),
                                               *upperdot, *lowerdot]
        dfout.loc[dataoutindex, "LastEdited"] = time.strftime("%Y-%m-%d %H:%M:%S")
        # Show Video stats on screen
        writedataline("~ VIDEO ~", 0.5)
        writedataline("Frame: {:05.0f}/{:.0f}".format(pos, capn), 1.5)
        writedataline("Time: {:04.0f}s/{:.0f}s".format(pos / capfps, capn / capfps), 2.5)
        writedataline("px per m: {:.3f}".format(scalebar * 10), 3.5, CYAN)
        verdict = dfout.loc[dataoutindex, "ScaleOK"]
        writedataline("ScaleOK: {}{}".format(verdict > 0, " (?)" if not (verdict in [0, 1]) else ""), 4.5, MAGENTA)
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
        writedataline("~ ENVDATA ~", 6, WHITE)
        writedataline("Time: {:04.0f}s/{:.0f}s".format(dataenvtime, max(dfenv["VideoTimeSecs"])), 7, WHITE)
        writedataline("Dist: {: 04.0f}.{:02.0f}m".format(dfenv["Dist"][dataenvindex],
                                                         abs(dfenv["Dist"][dataenvindex] * 100) % 100),
                      8, WHITE)
        writedataline("Salinity: {:.4f}".format(dfenv["Salinity"][dataenvindex]), 9, GREEN / 2)
        writedataline("Depth: {:.2f}m".format(dfenv["Depth"][dataenvindex]), 10, BLUE)
        writedataline("Temp: {:.4f}°C".format(dfenv["Temp"][dataenvindex]), 11, RED)
        dataport.blit(minigraph, (0, 420))
        pygame.draw.line(dataport, WHITE, [dfenv["Dist"][dataenvindex], 420], [dfenv["Dist"][dataenvindex], caph])
        pygame.draw.circle(dataport, WHITE, [int(dfenv["Dist"][dataenvindex]),
                                             int(scaledsalinity[dataenvindex]) + 420], 5)
        pygame.draw.circle(dataport, GREEN / 2, [int(dfenv["Dist"][dataenvindex]),
                                                 int(scaledsalinity[dataenvindex]) + 420], 3)
        pygame.draw.circle(dataport, WHITE, [int(dfenv["Dist"][dataenvindex]), int(scaledtemp[dataenvindex]) + 420], 5)
        pygame.draw.circle(dataport, RED, [int(dfenv["Dist"][dataenvindex]), int(scaledtemp[dataenvindex]) + 420], 3)
        pygame.draw.circle(dataport, WHITE, [int(dfenv["Dist"][dataenvindex]), int(scaleddepth[dataenvindex]) + 420], 5)
        pygame.draw.circle(dataport, BLUE, [int(dfenv["Dist"][dataenvindex]), int(scaleddepth[dataenvindex]) + 420], 3)
        # Draw in faunal records
        writedataline("~ FAUNA ~", 0.5, port=faunaport)
        barwidth = 0.5 * faunaport.get_width() / (len(faunanames) + 1)
        for i, name in enumerate(faunanames):
            selectedcol = WHITE if faunaindex == i else WHITE / 2
            writedataline("{} - {}: {:d} ({})".format(i + 1, name,
                                                      dfout.loc[dataoutindex, name],
                                                      sum(dfout.loc[:dataoutindex, name])),
                          i + 1.5, selectedcol, faunaport)
            pygame.draw.rect(faunaport, selectedcol,
                             [((2 * i) + 1) * barwidth, caph, barwidth, -2 * sum(dfout.loc[:dataoutindex, name])], 1)
            pygame.draw.rect(faunaport, selectedcol,
                             [((2 * i) + 1) * barwidth, caph, barwidth, -2 * dfout.loc[dataoutindex, name]])
            num = textfont.render(str(i + 1), True, selectedcol)
            faunaport.blit(num, [(((2 * i) + 1.5) * barwidth) - (num.get_width() / 2),
                                 caph - (2 * sum(dfout.loc[:dataoutindex, name])) - num.get_height()])
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
            elif e.type == MOUSEBUTTONDOWN:
                viewportmousepos = np.int32(pygame.mouse.get_pos()) - viewport.get_offset()
                if viewport.get_rect().collidepoint(viewportmousepos):
                    # Pauses if you click on the video
                    pause = not pause
                    # Line up playback position with multiple of skipspeed if skipping has occurred while paused
                    if pos == cap.get(cv2.CAP_PROP_POS_FRAMES):
                        pos, frame = getnewframe()
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

# https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
