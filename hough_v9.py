import random
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.feature import canny
from sklearn.cluster import KMeans
from matplotlib.figure import Figure
from sklearn.metrics import silhouette_score
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

###############

sil = []
sum_of_squared_distances = []
K = range(2, 24)
for k in K:
    kmeans = KMeans(n_clusters=k).fit(lines)
    sum_of_squared_distances.append(kmeans.inertia_)
    sil.append(silhouette_score(lines, kmeans.labels_, metric = 'euclidean'))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()

ax[0].plot(K, sum_of_squared_distances, 'bx-')
ax[0].set_xlabel('k')
ax[0].set_ylabel('Sum_of_squared_distances')
ax[0].set_title('Elbow Method For Optimal k')

ax[1].plot(K, sil, 'bx-')
ax[1].set_xlabel('k')
ax[1].set_ylabel('Silhouette_score')
ax[1].set_title('Silhouette Method For Optimal k')

plt.tight_layout()
plt.show()

optimal_k = sil.index(max(sil)) + 2

###############

kmeans = KMeans(n_clusters=optimal_k).fit(lines)

clusters = []
for j in range(optimal_k):
    clusters.append(lines[kmeans.labels_ == j])
    plt.scatter(lines[kmeans.labels_ == j][:, 0], lines[kmeans.labels_ == j][:, 1])
plt.show()

###############

new_clusters = []
for cluster in clusters:
    sil = []
    K = range(2, 5)
    for k in K:
        kmeans = KMeans(n_clusters=k).fit(cluster)        
        sil.append(silhouette_score(cluster, kmeans.labels_, metric = 'euclidean'))
    optimal_k = sil.index(max(sil)) + 2

    for j in range(optimal_k):
        new_clusters.append(cluster[kmeans.labels_ == j])
        plt.scatter(cluster[kmeans.labels_ == j][:, 0], cluster[kmeans.labels_ == j][:, 1])
plt.show()

################

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_xlim((0, image.shape[0]))
ax[0].set_ylim((0, image.shape[1]))

ax[1].imshow(label)

for cluster in new_clusters:
    m, b = cluster[0]

    # Plot it
    x_vals = np.array(ax[0].get_xlim())
    y_vals = b + m * x_vals
    ax[0].plot(x_vals, y_vals)

# show the plot
plt.tight_layout()
plt.show()

# For next week, try creating a regression line for the data and split the data from above the line to below the line for parallel lines.
# Could think about using a static variable to add to the regression intercept for the parallel lines. Then, only do kmeans clustering on
# the slops (m).

# As for movement, if all of the lines center are within the circle, then return the magnitude and direction of the difference of that and
# the center of the actual circle. If the lines center outside of the circle, then just returning direction should be sufficient enough.