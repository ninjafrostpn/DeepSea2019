import cv2
import numpy as np
import pygame
from pygame.locals import *
from scipy.ndimage.filters import convolve
import time

# Initialise window
pygame.init()

# Keys pressed in the frame just shown...
keys = set()
# ...and the one before that (for good edge detection)
prevkeys = set()

# Colours
WHITE = np.int32([255] * 3)

# Font for drawing text with
textfont = pygame.font.Font(None, 50)

# Initialise all window and video-getting tools
cap = cv2.VideoCapture("IsisROV_dive148_TRANSECT.m4v")
capn = cap.get(cv2.CAP_PROP_FRAME_COUNT)
capfps = cap.get(cv2.CAP_PROP_FPS)
capw, caph = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
capsize = np.int32([capw, caph])
screensize = np.int32([capw, caph + 50])
screen = pygame.display.set_mode(screensize)
viewport = screen.subsurface([0, 50, capw, caph])
pygame.display.set_caption("VIDEO", "Video")
print("{:.0f} frames ({:.0f} x {:.0f}) at {} FPS".format(capn, capw, caph, capfps))


def getnewframe(pos=None):
    if pos is not None:
        if pos < 0:
            print(pos)
            pos = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ret, frame = cap.read()
    if not ret:
        print("Oh No")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(np.fliplr(frame))
    return pos, frame


# Blob-shaped filter used to find red blobs
redblobfinder = - 5 * np.ones([17, 17])
redblobfinder = cv2.circle(redblobfinder, (8, 8), 6, 10, -1)

pause = False
frame = getnewframe()
pos = cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
skipspeed = 5

while True:
    if K_SPACE in keys and K_SPACE not in prevkeys:
        pause = not pause
    if K_RIGHT in keys:
        if not pause:
            pos += 10 * skipspeed
            pos, frame = getnewframe(pos)
        elif K_RIGHT not in prevkeys:
            pos += skipspeed
            pos, frame = getnewframe(pos)
    elif K_LEFT in keys:
        if not pause:
            pos -= 10 * skipspeed
            pos, frame = getnewframe(pos)
        elif K_LEFT not in prevkeys:
            print(pause)
            pos -= skipspeed
            pos, frame = getnewframe(pos)
    elif not pause:
        # elif, not if, to provide allow skipping back to frame 0
        for i in range(skipspeed):
            pos, frame = getnewframe()
    screen.fill(0)
    pframe = pygame.surfarray.make_surface(frame)
    # Look for the red dots in the "middle half" of the image area
    redness = np.sum(frame[330:700, 144:432] * [1, -0.5, -0.5], axis=2)
    dotlikeness = convolve(redness, redblobfinder, mode="constant", cval=0)
    # First-guess upper and lower dots' positions
    upperdot = np.int32(np.unravel_index(np.argmax(dotlikeness[:, :144]), (700, 144))) + (330, 144)
    lowerdot = np.int32(np.unravel_index(np.argmax(dotlikeness[:, 144:]), (700, 144))) + (330, 288)
    # Draw video frame to screen
    viewport.blit(pframe, (0, 0))
    # Draw dot-finder reticle
    pygame.draw.rect(viewport, WHITE, [330, 144, 370, 288], 1)
    pygame.draw.line(viewport, WHITE, [330, 288], [700, 288], 1)
    pygame.draw.circle(viewport, WHITE, upperdot, 20, 1)
    pygame.draw.circle(viewport, WHITE, lowerdot, 20, 1)
    pygame.draw.line(viewport, WHITE, lowerdot, upperdot, 1)
    screen.blit(textfont.render("Frame: {:07.0f}".format(pos), True, WHITE), (0, 0))
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

"""
https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
"""
