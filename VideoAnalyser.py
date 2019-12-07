import cv2
import numpy as np
import pygame
from pygame.locals import *
from scipy.ndimage.filters import convolve
import time

pygame.init()
keys = set()
WHITE = np.int32([255] * 3)

textfont = pygame.font.Font(None, 50)

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

# Blob-shaped filter used to find red blobs
redblobfinder = - 5 * np.ones([17, 17])
redblobfinder = cv2.circle(redblobfinder, (8, 8), 6, 10, -1)

pause = False
pos = cap.set(cv2.CAP_PROP_POS_FRAMES, 15500)
skipspeed = 5

while True:
    if not pause:
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    for i in range(skipspeed):
        ret, frame = cap.read()
    if ret:
        screen.fill(0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(np.fliplr(frame))
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
