import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.feature import canny
from sklearn.cluster import KMeans
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage.transform import probabilistic_hough_line, hough_line, hough_line_peaks

# Line finding using the Probabilistic Hough Transform
image = cv2.imread("data/train/samples/CBED_1.tif", cv2.IMREAD_GRAYSCALE)
label = cv2.imread("data/train/labels/CBED_1.png", cv2.IMREAD_GRAYSCALE)

# Get probabilistic hough transform
edges = canny(image, 2, 1, 25)
out = probabilistic_hough_line(edges, threshold=250, line_length=10, line_gap=3)

fig = Figure(figsize=(image.shape[0]/120, image.shape[1]/120), dpi=120)
canvas = FigureCanvas(fig)
ax = fig.gca()

ax.imshow(edges * 0, cmap=cm.gray)
lines = []
for line in out:
    p0, p1 = line    
    # ax.plot((p0[0], p1[0]), (p0[1], p1[1]), c='0.5')
    
    # Extrapolate line
    m = (p0[1] - p1[1]) / (p0[0] - p1[0])
    b = p0[1] - (m * p0[0])

    # Append it to lines output
    lines.append([m, b])

# Perform KMeans
kmeans = KMeans(n_clusters=12, random_state=0).fit(lines) # 6 bands

# Plot lines
for line in lines:
    m, b = line

    # Plot it
    x_vals = np.array(ax.get_xlim())
    y_vals = b + m * x_vals
    ax.plot(x_vals, y_vals, c='0.5')

for line in kmeans.cluster_centers_:
    m, b = line

    # Plot it
    x_vals = np.array(ax.get_xlim())
    y_vals = b + m * x_vals
    ax.plot(x_vals, y_vals, c='0.5')

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

# Try eroding
# kernel = np.ones((3, 3),np.uint8)
# image_from_plot = cv2.dilate(image_from_plot, kernel, iterations = 1)
# image_from_plot = cv2.erode(image_from_plot, kernel, iterations = 1)

# Regular hough transform
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
h, theta, d = hough_line(image_from_plot, theta=tested_angles)

# Generate plots
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(image_from_plot, cmap=cm.gray)
ax[1].set_title('Extrapolated probabilistic hough transform')
ax[1].set_axis_off()

angle_step = 0.5 * np.diff(theta).mean()
d_step = 0.5 * np.diff(d).mean()
bounds = [np.rad2deg(theta[0] - angle_step),
          np.rad2deg(theta[-1] + angle_step),
          d[-1] + d_step, d[0] - d_step]
ax[2].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
ax[2].set_title('Hough space')
ax[2].set_xlabel('Angles (degrees)')
ax[2].set_ylabel('Distance (pixels)')
ax[2].axis('image')

plt.tight_layout()
plt.show()

# Generate output plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image_from_plot, cmap=cm.gray)
ax[0].set_ylim((image_from_plot.shape[0], 0))
ax[0].set_axis_off()
ax[0].set_title('Hough transform applied on probabilistic hough transform')

for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    ax[0].axline((x0, y0), slope=np.tan(angle + np.pi/2))

ax[1].imshow(image, cmap=cm.gray)
ax[1].set_ylim((image.shape[0], 0))
ax[1].set_axis_off()
ax[1].set_title('Detected lines')

for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    ax[1].axline((x0, y0), slope=np.tan(angle + np.pi/2))

plt.tight_layout()
plt.show()