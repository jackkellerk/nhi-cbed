import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.feature import canny
from sklearn.cluster import KMeans
from matplotlib.figure import Figure
from skimage.transform import probabilistic_hough_line
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# COMA Free Alignment; 5 epochs; Deep hough transform for semantic line detection -- cnn paper

for j in range(1, 21):
    # Line finding using the Probabilistic Hough Transform
    image = cv2.imread("data/train/samples/CBED_" + str(j) + ".tif", cv2.IMREAD_GRAYSCALE)
    label = cv2.imread("data/train/labels/CBED_" + str(j) + ".png", cv2.IMREAD_ANYCOLOR)
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

    # Generate space for clustered
    fig = Figure(figsize=(image.shape[0]/120, image.shape[1]/120), dpi=120)
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.imshow(edges * 0, cmap=cm.gray)

    # Perform KMeans
    n = 12
    kmeans = KMeans(n_clusters=n).fit(lines)

    # Plot lines
    for line in kmeans.cluster_centers_:
        m, b = line

        # Plot it
        x_vals = np.array(ax.get_xlim())
        y_vals = b + m * x_vals
        ax.plot(x_vals, y_vals, 'r-')

    for i in range(len(lines)):
        if i % 46 == 0:
            m, b = lines[i]

            # Plot it
            x_vals = np.array(ax.get_xlim())
            y_vals = b + m * x_vals
            ax.plot(x_vals, y_vals, 'r-')

    ax.set_xlim((0, image.shape[0]))
    ax.set_ylim((0, image.shape[1]))
    ax.axis('off')

    canvas.draw()
    image_from_plot_clustered = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot_clustered = image_from_plot_clustered.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_from_plot_clustered = np.flipud(image_from_plot_clustered)
    image_from_plot_clustered = cv2.cvtColor(image_from_plot_clustered, cv2.COLOR_BGR2GRAY)
    image_from_plot_clustered = np.where(image_from_plot_clustered == 255, 0, image_from_plot_clustered)
    image_from_plot_clustered = np.where(image_from_plot_clustered != 0, 255, image_from_plot_clustered)

    # Generate space for iterated
    fig = Figure(figsize=(image.shape[0]/120, image.shape[1]/120), dpi=120)
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.imshow(edges * 0, cmap=cm.gray)

    # Plot lines
    for i in range(len(lines)):
        if i % 46 == 0:
            m, b = lines[i]

            # Plot it
            x_vals = np.array(ax.get_xlim())
            y_vals = b + m * x_vals
            ax.plot(x_vals, y_vals, 'r-')

    ax.set_xlim((0, image.shape[0]))
    ax.set_ylim((0, image.shape[1]))
    ax.axis('off')

    canvas.draw()
    image_from_plot_iter = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot_iter = image_from_plot_iter.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_from_plot_iter = np.flipud(image_from_plot_iter)
    image_from_plot_iter = cv2.cvtColor(image_from_plot_iter, cv2.COLOR_BGR2GRAY)
    image_from_plot_iter = np.where(image_from_plot_iter == 255, 0, image_from_plot_iter)
    image_from_plot_iter = np.where(image_from_plot_iter != 0, 255, image_from_plot_iter)

    # Generate plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(image_from_plot_extrapolated, cmap=cm.gray)
    ax[1].set_title('Extrapolated lines')
    ax[1].set_axis_off()

    ax[2].imshow(label, cmap=cm.gray)
    ax[2].set_title('Label')
    ax[2].set_axis_off()

    plt.tight_layout()
    plt.savefig("./outputimages/" + str(j) + ".png")