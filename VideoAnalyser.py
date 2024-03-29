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
toprowkeys = [K_q, K_w, K_e, K_r, K_t, K_y]

# Colours
WHITE = np.int32([255] * 3)
GREEN = np.int32([0, 255, 0])
CYAN = np.int32([0, 255, 255])
MAGENTA = np.int32([255, 0, 255])
RED = np.int32([255, 0, 0])
BLUE = np.int32([0, 0, 255])

# Font for drawing text with
textfont = pygame.font.Font(None, 25)

# Initialise all window and video-getting tools
cap = cv2.VideoCapture("IsisROV_dive148_TRANSECT.m4v")
capn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
capfps = cap.get(cv2.CAP_PROP_FPS)
capw, caph = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
capsize = np.int32([capw, caph])
screensize = np.int32([capw + 600, caph])
screen = pygame.display.set_mode(screensize)
dataport = screen.subsurface([0, 0, 300, caph])
viewport = screen.subsurface([300, 0, capw, caph])
faunaport = screen.subsurface([capw + 300, 0, 300, caph])
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
    port.blit(textfont.render(text, True, col), (20, 25 * lineno))


# Blob-shaped filter used to find red blobs
redblobfinder = - 5 * np.ones([17, 17])
redblobfinder = cv2.circle(redblobfinder, (8, 8), 6, 10, -1)

# Set initial state of interface
pause = True
showreticle = True
startpos = 55500
skipspeed = int(30 * capfps)  # A frame every 30 sec
pos, frame = getnewframe(startpos)
cap.set(cv2.CAP_PROP_POS_FRAMES, startpos)
faunaindex = 0
faunaoffset = 0
coverindex = 0
# If 0, counting species, 1, counting habitat representation
countmode = 0
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
# The Cover columns give the number of 16ths of the image in which the bottom type is present
scalestats = ["ScalePX", "ScalePX/m", "AreaEstimate", "UScaleX", "UScaleY", "LScaleX", "LScaleY"]
faunanames = ["Shreemp", "Extendoworm", "Clamaybe", "ChonkAnemone", "MudAnemone", "Ophi",
              "Feesh", "BigSqueed", "SmolSqueed", "Blueb", "TinyWhite", "SeaSosig", "UrChin"]
covertypes = ["HardBtm", "SoftBtm", "Clams", "Bacteria", "Query", "Unusable"]
# For ScaleOK, -1 represents unchecked, -2 or 2 represent bad/good autocheck, 0 or 1 represent confirmed bad/good
# For Done, 0 represents incomplete, 1 represents complete for count check, lasers check and cover check
coldefaults = {"Frame": np.arange(capn), **{col: np.nan for col in dfenv.columns},
               "ScaleOK": -1, **{i: np.nan for i in scalestats},
               **{name: 0 for name in faunanames},
               **{covertype: 0 for covertype in covertypes},
               "Done": 0, "LastEdited": "nan"}
try:
    # Filenames are of format "AllData-Z.csv", where Z is the version nr
    csvnames = sorted(glob("AllData-*.csv"), key=getfileno)
    if len(csvnames) == 0:
        csvinname = "Matching Files"
        raise FileNotFoundError
    csvinname = csvnames[-1]
    dfout = pd.read_csv(csvinname)
    csvoutname = "AllData-{}.csv".format(getfileno(csvinname) + 1)
    coldefaultskeys = np.object_(list(coldefaults.keys()))
    colsinmask = np.isin(coldefaultskeys, dfout.columns)
    if not np.all(colsinmask):
        print(coldefaultskeys[~colsinmask], "not found, adding")
        dfout = dfout.assign(**{key: coldefaults[key] for key in coldefaultskeys[~colsinmask]})
    print("Loaded in", csvinname)
except FileNotFoundError:
    csvoutname = "AllData-0.csv"
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
            countmode = 0
            # Number keys select that numbered faunal record
            faunaindex = numberkeys.index(numberspressed[0])
        toprowkeyspressed = list(keys.intersection(toprowkeys))
        if len(toprowkeyspressed) > 0:
            countmode = 1
            # Letter keys on top row select that lettered bottom type record
            coverindex = toprowkeys.index(toprowkeyspressed[0])
        if K_COMMA in keys and K_COMMA not in prevkeys:
            # Comma key to move fauna selector numbers up (because it has the < symbol on it)
            if faunaoffset - 9 >= 0:
                faunaoffset -= 9
        if K_PERIOD in keys and K_PERIOD not in prevkeys:
            # Comma key to move fauna selector numbers down (because it has the > symbol on it)
            if faunaoffset + 9 < len(faunanames):
                faunaoffset += 9
        if countmode == 0:
            if 0 <= faunaindex + faunaoffset < len(faunanames):
                # These two are placed here (before update of dataoutindex)so that they get the previous frame
                # (when the key was pressed) before the next one is drawn
                if K_UP in keys and K_UP not in prevkeys:
                    # Up key adds a count for the selected faunal record
                    dfout.loc[dataoutindex, faunanames[faunaindex + faunaoffset]] += 1
                    dfout.loc[dataoutindex, "LastEdited"] = time.strftime("%Y-%m-%d %H:%M:%S")
                if K_DOWN in keys and K_DOWN not in prevkeys:
                    # Up key removes a count for the selected faunal record
                    dfout.loc[dataoutindex, faunanames[faunaindex + faunaoffset]] -= 1
                    dfout.loc[dataoutindex, "LastEdited"] = time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            covertype = covertypes[coverindex]
            if K_UP in keys and K_UP not in prevkeys:
                dfout.loc[dataoutindex, covertype] += 1
                dfout.loc[dataoutindex, "LastEdited"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if K_DOWN in keys and K_DOWN not in prevkeys:
                dfout.loc[dataoutindex, covertype] -= 1
                dfout.loc[dataoutindex, "LastEdited"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if K_PAGEUP in keys and K_PAGEUP not in prevkeys:
                dfout.loc[dataoutindex, covertype] = 16
                dfout.loc[dataoutindex, "LastEdited"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if K_PAGEDOWN in keys and K_PAGEDOWN not in prevkeys:
                dfout.loc[dataoutindex, covertype] = 0
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
        if K_RETURN in keys and K_RETURN not in prevkeys:
            # Enter to register/deregister count completeness
            oldverdict = dfout.loc[dataoutindex, "Done"]
            if oldverdict == 0:
                dfout.loc[dataoutindex, "Done"] = 1
            else:
                dfout.loc[dataoutindex, "Done"] = 0
            dfout.loc[dataoutindex, "LastEdited"] = time.strftime("%Y-%m-%d %H:%M:%S")
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
        elif K_END in keys:
            pos = capn - 1
            pos, frame = getnewframe(pos)
        elif K_HOME in keys:
            pos = 0
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
            # Draw cover-apportioning grid
            for i in range(1, 4):
                pygame.draw.line(viewport, GREEN, [10 + ((capw - 20) * (i/4)), 0], [10 + ((capw - 20) * (i/4)), caph])
                pygame.draw.line(viewport, GREEN, [10, caph * (i / 4)], [capw - 10, caph * (i / 4)])
            # Draw dot-finder reticle
            pygame.draw.rect(viewport, WHITE, [330, 144, 370, 288], 2)
            pygame.draw.line(viewport, WHITE, [330, 288], [700, 288], 2)
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
            if oldverdict in [-2, -1, 2]:
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
        writedataline("FrameSkip: {} ({:.0f}:{:.0f})".format(skipspeed,
                                                             pos // skipspeed,
                                                             np.ceil((capn - pos) / skipspeed)),
                      5.5)
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
        writedataline("~ ENVDATA ~", 7, WHITE)
        writedataline("Time: {:04.0f}s/{:.0f}s".format(dataenvtime, max(dfenv["VideoTimeSecs"])), 8, WHITE)
        writedataline("Dist: {: 04.0f}.{:02.0f}m".format(dfenv["Dist"][dataenvindex],
                                                         abs(dfenv["Dist"][dataenvindex] * 100) % 100),
                      9, WHITE)
        writedataline("Salinity: {:.4f}".format(dfenv["Salinity"][dataenvindex]), 10, GREEN / 2)
        writedataline("Depth: {:.2f}m".format(dfenv["Depth"][dataenvindex]), 11, BLUE)
        writedataline("Temp: {:.4f}°C".format(dfenv["Temp"][dataenvindex]), 12, RED)
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
        writedataline("Done: {}".format("YES" if dfout.loc[dataoutindex, "Done"] else "NO"),
                      1.5, port=faunaport)
        for i, covertype in enumerate(covertypes):
            selectedcol = GREEN if (coverindex == i) and (countmode == 1) else GREEN / 2
            writedataline("{} - {}: {}".format(chr(toprowkeys[i]), covertype,
                                               dfout.loc[dataoutindex, covertype]),
                          i + 2.5, selectedcol, faunaport)
        barwidth = 0.5 * faunaport.get_width() / (min(len(faunanames), 9) + 1)
        for i, name in enumerate(faunanames[faunaoffset:faunaoffset + 9]):
            selectedcol = WHITE if (faunaindex == i) and (countmode == 0) else WHITE / 2
            writedataline("{} - {}: {} ({})".format(i + 1, name,
                                                    dfout.loc[dataoutindex, name],
                                                    sum(dfout.loc[:dataoutindex, name])),
                          i + 2.5 + len(covertypes), selectedcol, faunaport)
            pygame.draw.rect(faunaport, selectedcol,
                             [((2 * i) + 1) * barwidth, caph, barwidth, -sum(dfout.loc[:dataoutindex, name])], 1)
            pygame.draw.rect(faunaport, selectedcol,
                             [((2 * i) + 1) * barwidth, caph, barwidth, -dfout.loc[dataoutindex, name]])
            num = textfont.render(str(i + 1), True, selectedcol)
            faunaport.blit(num, [(((2 * i) + 1.5) * barwidth) - (num.get_width() / 2),
                                 caph - sum(dfout.loc[:dataoutindex, name]) - num.get_height()])
        writedataline("{} {}".format("<<" if faunaoffset > 0 else "  ",
                                     ">>" if faunaoffset + 9 < len(faunanames) else "  "),
                      2.5 + min(len(faunanames), 9) + len(covertypes), WHITE, faunaport)
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
                         columns=["Frame", "VideoTimeSecs", "VideoTime", "ActualTime", "Dist",
                                  "Lat", "Lon",
                                  *faunanames, *covertypes,
                                  "Depth", "Temp", "Salinity",
                                  *scalestats, "ScaleOK",
                                  "Done", "LastEdited"])
            unsaved = False
        except PermissionError:
            print(csvoutname, "might be open elsewhere, waiting for you to close it...")
            time.sleep(5)
    print("Save completed")
    print("Exited on frame", int(pos))

# https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
