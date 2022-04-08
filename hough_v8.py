import random
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.feature import canny
from sklearn.cluster import KMeans
from matplotlib.figure import Figure
from sklearn.cluster import AffinityPropagation
from skimage.transform import probabilistic_hough_line
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Line finding using the Probabilistic Hough Transform
image = cv2.imread("data/train/samples/CBED_1.tif", cv2.IMREAD_GRAYSCALE)
label = cv2.imread("data/train/labels/CBED_1.png", cv2.IMREAD_ANYCOLOR)
label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

# Get probabilistic hough transform
edges = canny(image, 2, 1, 25)
out = probabilistic_hough_line(edges, threshold=250, line_length=10, line_gap=3)

# Generate space for probabilistic
fig = Figure(figsize=(image.shape[0]/120, image.shape[1]/120), dpi=120)
canvas = FigureCanvas(fig)
ax = fig.gca()

ax.imshow(edges * 0, cmap=cm.gray)
lines = []
for line in out:
    p0, p1 = line    
    ax.plot((p0[0], p1[0]), (p0[1], p1[1]), c='0.5')
    
    # Extrapolate line
    m = (p0[1] - p1[1]) / (p0[0] - p1[0])
    b = p0[1] - (m * p0[0])

    # Append it to lines output
    lines.append([m, b])
lines = np.array(lines)
lines = lines[np.argsort(lines[:, 0])]

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

# Generate space for extrapolated hough
fig = Figure(figsize=(image.shape[0]/120, image.shape[1]/120), dpi=120)
canvas = FigureCanvas(fig)
ax = fig.gca()

ax.imshow(edges * 0, cmap=cm.gray)

# Plot lines
for line in lines:
    m, b = line

    # Plot it
    x_vals = np.array(ax.get_xlim())
    y_vals = b + m * x_vals
    ax.plot(x_vals, y_vals, c='0.5')

ax.set_xlim((0, image.shape[0]))
ax.set_ylim((0, image.shape[1]))
ax.axis('off')

canvas.draw()
image_from_plot_extrapolated = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
image_from_plot_extrapolated = image_from_plot_extrapolated.reshape(fig.canvas.get_width_height()[::-1] + (3,))
image_from_plot_extrapolated = np.flipud(image_from_plot_extrapolated)
image_from_plot_extrapolated = cv2.cvtColor(image_from_plot_extrapolated, cv2.COLOR_BGR2GRAY)
image_from_plot_extrapolated = np.where(image_from_plot_extrapolated == 255, 0, image_from_plot_extrapolated)
image_from_plot_extrapolated = np.where(image_from_plot_extrapolated != 0, 255, image_from_plot_extrapolated)

# Generate plots
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(image_from_plot, cmap=cm.gray)
ax[1].set_title('Probabilistic hough transform')
ax[1].set_axis_off()

ax[2].imshow(image_from_plot_extrapolated, cmap=cm.gray)
ax[2].set_title('Extrapolated lines')
ax[2].set_axis_off()

plt.tight_layout()
plt.show()

############

num_clusters = 6
plt.scatter(lines[:, 0], lines[:, 1])

for i in range(1, num_clusters):
    spacing = (abs(np.max(lines[:, 0])) + abs(np.min(lines[:, 0]))) / num_clusters
    plt.axvline(x=(np.min(lines[:, 0]) + i * spacing))

plt.show()

############

spacing = (abs(np.max(lines[:, 0])) + abs(np.min(lines[:, 0]))) / num_clusters

clusters = [[] for i in range(num_clusters)]
for i in range(1, num_clusters + 1):
    separated_lines = []

    for line in lines:
        if i == 1:
            if line[0] <= (np.min(lines[:, 0]) + (i * spacing)):
                separated_lines.append([line[0], line[1]])
        elif i == num_clusters:
            if line[0] > (np.min(lines[:, 0]) + ((i - 1) * spacing)):
                separated_lines.append([line[0], line[1]])
        else:            
            if line[0] <= (np.min(lines[:, 0]) + (i * spacing)) and line[0] > (np.min(lines[:, 0]) + ((i - 1) * spacing)):
                separated_lines.append([line[0], line[1]])

    clusters[i - 1] = np.array(separated_lines)
clusters = np.array(clusters)

############

for i in range(num_clusters):
    plt.scatter(clusters[i][:, 0], clusters[i][:, 1])
    
    if i < num_clusters - 1:
        plt.axvline(x=(np.min(lines[:, 0]) + ((i + 1) * spacing)))

plt.show()

###########

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()

ax[1].imshow(image, cmap=cm.gray)
ax[1].set_xlim((0, image.shape[0]))
ax[1].set_ylim((0, image.shape[1]))

for i in range(num_clusters):
    cluster = np.array(clusters[i])    

    # Assume 2 groups
    k = 2
    kmeans = KMeans(n_clusters=k).fit(cluster)    

    colors = ["r", "b", "g", "c", "m", "y"]
    color1 = random.choice(colors)
    colors.remove(color1)
    color2 = random.choice(colors)

    for j in range(cluster.shape[0]):
        ax[0].plot(cluster[j][0], cluster[j][1], (color1 + "o") if kmeans.labels_[j] == 0 else (color2 + "o"))

    if i < num_clusters - 1:
        ax[0].axvline(x=(np.min(lines[:, 0]) + ((i + 1) * spacing)))

    for j in range(k):
        m, b = cluster[kmeans.labels_ == j][0]

        # Plot it
        x_vals = np.array(ax[1].get_xlim())
        y_vals = b + m * x_vals
        ax[1].plot(x_vals, y_vals, (color1 + "-") if j == 0 else (color2 + "-"))

# show the plot
plt.tight_layout()
plt.show()