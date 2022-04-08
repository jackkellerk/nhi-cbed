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
image = cv2.imread("data/train/samples/CBED_11.tif", cv2.IMREAD_GRAYSCALE)
label = cv2.imread("data/train/labels/CBED_11.png", cv2.IMREAD_ANYCOLOR)
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

###################

colors = ['green', 'royalblue', 'orange', 'firebrick', 'teal', 'mediumpurple', 'deeppink', 'lawngreen', 'sienna']

# Do k-means on just the x-axis
x_points = lines[:, 0].reshape(-1,1)

sil = []
sum_of_squared_distances = []
K = range(2, 24)
for k in K:
    kmeans = KMeans(n_clusters=k).fit(x_points)
    sum_of_squared_distances.append(kmeans.inertia_)
    sil.append(silhouette_score(x_points, kmeans.labels_, metric = 'euclidean'))

optimal_k = sil.index(max(sil)) + 2

####################

# Plot optimal k

kmeans = KMeans(n_clusters=optimal_k).fit(x_points)

clusters = []
for j in range(optimal_k):
    clusters.append(x_points[kmeans.labels_ == j])
    plt.scatter(x_points[kmeans.labels_ == j], [0 for i in range(len(x_points[kmeans.labels_ == j]))], c=colors[j])
plt.show()

####################

# Do linear regression

clusters = []
for j in range(optimal_k):
    clusters.append(lines[kmeans.labels_ == j])
    plt.scatter(lines[kmeans.labels_ == j][:, 0], lines[kmeans.labels_ == j][:, 1], c=colors[j])

from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(lines[:, 0].reshape(-1,1), lines[:, 1])
m = lr.coef_
b = lr.intercept_

stds = []
means = []
for j in range(optimal_k):
    stds.append(np.std(lines[kmeans.labels_ == j][:, 1]))
    means.append(np.mean(lines[kmeans.labels_ == j][:, 0]))

x_vals = np.array(plt.gca().get_xlim())
y_vals = b + m * x_vals
plt.plot(x_vals, y_vals, c='0.5')
plt.show()

####################

# Plot lines on graph

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_xlim((0, image.shape[0]))
ax[0].set_ylim((0, image.shape[1]))

ax[1].imshow(image, cmap=cm.gray)
ax[1].set_xlim((0, image.shape[0]))
ax[1].set_ylim((0, image.shape[1]))

for line in lines:
    m, b = line
    x_vals = np.array(ax[0].get_xlim())
    y_vals = b + m * x_vals
    ax[0].plot(x_vals, y_vals, 'r')

for j in range(optimal_k):
    m = means[j]
    b1 = lr.predict(m.reshape(-1, 1)) + stds[j]
    b2 = lr.predict(m.reshape(-1, 1)) - stds[j]

    x_vals = np.array(ax[1].get_xlim())
    y_vals = b1 + m * x_vals
    ax[1].plot(x_vals, y_vals, c=colors[j])
    
    y_vals = b2 + m * x_vals
    ax[1].plot(x_vals, y_vals, c=colors[j])

plt.show()