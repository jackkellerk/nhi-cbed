import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage import data
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.transform import probabilistic_hough_line

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Line finding using the Probabilistic Hough Transform
image = cv2.imread("data/train/samples/CBED_1.tif", cv2.IMREAD_GRAYSCALE)
label = cv2.imread("data/train/labels/CBED_1.png")
edges = canny(image, 2, 1, 25)
lines = probabilistic_hough_line(edges, threshold=250, line_length=10, line_gap=3)

# Generating figure 2
fig = Figure(figsize=(image.shape[0]/120, image.shape[1]/120), dpi=120)
canvas = FigureCanvas(fig)
ax = fig.gca()

ax.imshow(edges * 0, cmap=cm.gray)
for line in lines:
    p0, p1 = line
    ax.plot((p0[0], p1[0]), (p0[1], p1[1]), c='0.5')
ax.set_xlim((0, image.shape[0]))
ax.set_ylim((0, image.shape[1]))
ax.axis('off')

canvas.draw()
image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
image_from_plot = np.flipud(image_from_plot)
image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_BGR2GRAY)
image_from_plot = np.where(image_from_plot == 255, 0, image_from_plot)
image_from_plot = np.where(image_from_plot != 0, 255, image_from_plot)
print(image_from_plot.shape)
cv2.imwrite("test.png", image_from_plot)

## TODO: Fix
from skimage import data, filters
fig, ax = plt.subplots(nrows=1, ncols=3)

image = image_from_plot
edges = filters.sobel(image)

low = 0.1
high = 0.75

lowt = (edges > low).astype(int)
hight = (edges > high).astype(int)
hyst = filters.apply_hysteresis_threshold(edges, low, high)

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original image')

ax[1].imshow(lowt, cmap='magma')
ax[1].set_title('Low threshold')

ax[2].imshow(hight + hyst, cmap='magma')
ax[2].set_title('Hysteresis threshold')

for a in ax.ravel():
    a.axis('off')

plt.tight_layout()

plt.show()

# TODO: Next week, run hough transform on top of probabilistic hough transform binary image
# TODO: Next week, have lines intensity correspond to probability. Then, rerun hysteresis thresholding
# TODO: Also, start working on a way to determine a score for how well my method works compared to labels. After that, hyperparameter analysis