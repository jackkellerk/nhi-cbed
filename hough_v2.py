import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage import data
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.transform import probabilistic_hough_line

# Line finding using the Probabilistic Hough Transform
image = cv2.imread("data/train/samples/CBED_1.tif", cv2.IMREAD_GRAYSCALE)
label = cv2.imread("data/train/labels/CBED_1.png")
edges = canny(image, 2, 1, 25)
lines = probabilistic_hough_line(edges, threshold=250, line_length=10, line_gap=3)

# Generating figure 2
fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(edges * 0, cmap=cm.gray)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]), c='1')
ax[2].set_xlim((0, image.shape[0]))
ax[2].set_ylim((0, image.shape[1]))
ax[2].set_title('Probabilistic Hough')

ax[3].imshow(label[:,:,::-1])
ax[3].set_title('Input image labels')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()