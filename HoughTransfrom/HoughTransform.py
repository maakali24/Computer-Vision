import cv2
import numpy as np
from skimage import io

# read image from disk
myImage = cv2.imread('Maak.png')
# convert BGR to gray scale
gray = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
# find edges using canny edge detector
cannyEdges = cv2.Canny(gray, 100, 200)
# find lines in image
lines = cv2.HoughLines(cannyEdges, 1, np.pi / 180, 200)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(myImage, (x1, y1), (x2, y2), (0, 0, 255,), 2)

# show result
cv2.imshow('Canny Edges', cannyEdges)
cv2.imshow('HT lines', myImage)

# save result
io.imsave('result/Canny edges.png', cannyEdges)
io.imsave('result/HT lines.png', myImage)

# harris Corner detector
gray = np.float32(gray)
harrisCorner = cv2.cornerHarris(gray, 4, 7, 0, 0.04)

# result is dilated for marking the corners,not important
dilated = cv2.dilate(harrisCorner, None)

# Threshold for an optimal value, it may vary depending on the image.
myImage[dilated > 0.01 * dilated.max()] = [0, 0, 255]

# show result
cv2.imshow('Hough Transform lines & harris corners', myImage)
cv2.imshow('Harris Corners (With out Dilated)', harrisCorner)
# save result in disk
io.imsave('result/Harris Corner detection & HT lines.png', myImage)
io.imsave('result/Harris Corners (With out Dilated).png', harrisCorner)

# waiting for key
k = cv2.waitKey(0)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
