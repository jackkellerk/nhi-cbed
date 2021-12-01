import numpy as np
import sys
from skimage.transform import probabilistic_hough_line, hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage import data
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def hough_transform_array(image, threshold=250, line_length=10, line_gap=3):
    edges = canny(image, 2, 1, 25)
    lines = probabilistic_hough_line(edges, threshold=threshold, line_length=line_length, line_gap=line_gap)

    fig = Figure(figsize=(image.shape[0]/120, image.shape[1]/120), dpi=120)
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.imshow(edges * 0, cmap=cm.gray)
    slopes = []
    for line in lines:
        p0, p1 = line
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]), c='0.5')

        if p1[0] - p0[1] != 0:
            slopes.append((p1[1] - p0[1]) / (p1[0] - p0[1]))    
    ax.set_xlim((0, image.shape[0]))
    ax.set_ylim((0, image.shape[1]))
    ax.axis('off')

    # Slopes stuff
    slopes = np.array(slopes).reshape(-1, 1)
    kmeans = KMeans(n_clusters=6, random_state=0).fit(slopes) # 6 bands
    if display_slopes:
        print(kmeans.cluster_centers_)

    canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_from_plot = np.flipud(image_from_plot)
    image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_BGR2GRAY)
    image_from_plot = np.where(image_from_plot == 255, 0, image_from_plot)
    image_from_plot = np.where(image_from_plot != 0, 255, image_from_plot)
    
    return image_from_plot

def accuracy(image, label):
    total = 0
    label_total = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if label[i][j] > 0:
                label_total += 1
            
            if image[i][j] > 0 and label[i][j] > 0:
                total += 1

    return (total / label_total) * 100

def rmse(image, label):
    new_image = []
    new_label = []

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if label[i][j] > 0 or image[i][j] > 0:
                new_image.append(image[i][j])
                new_label.append(label[i][j])
    
    return mean_squared_error(new_label, new_image, squared=False)

# Get command line opts
display_graphs = True if "-d" in sys.argv[1:] else False
display_sift = True if "-s" in sys.argv[1:] else False
display_slopes = True if "-sl" in sys.argv[1:] else False
print("Display graphs: " + str(display_graphs))
print("Display slopes: " + str(display_slopes))
print("Display SIFT: " + str(display_sift))

# Loop over every item in the dataset
for i in range(1, 21):
    # Line finding using the Probabilistic Hough Transform
    image = cv2.imread("data/train/samples/CBED_" + str(i) + ".tif", cv2.IMREAD_GRAYSCALE)
    label = cv2.imread("data/train/labels/CBED_" + str(i) + ".png", cv2.IMREAD_GRAYSCALE)

    # Generate hough transform for image and label
    image_from_plot = hough_transform_array(image)    
    label_from_plot = hough_transform_array(label)    

    image_from_plot_2 = hough_transform_array(image_from_plot, threshold=150, line_length=5, line_gap=1)

    # Print accuracy of the method
    print("\n\n")
    print("CBED_" + str(i) + ".tif")
    print("Hough Transform:")
    print("RMSE: " + str(rmse(image_from_plot, label_from_plot)))
    print("Accuracy: " + str(accuracy(image_from_plot, label_from_plot)) + "%")
    print("\nHough Transform^2:")
    print("RMSE: " + str(rmse(image_from_plot_2, label_from_plot)))
    print("Accuracy: " + str(accuracy(image_from_plot_2, label_from_plot)) + "%")

    # Generate graph
    if display_graphs:
        fig, ax = plt.subplots(nrows=1, ncols=3)

        ax[0].imshow(image_from_plot, cmap='gray')
        ax[0].set_title('Hough Transform')
        
        ax[1].imshow(image_from_plot_2, cmap='gray')
        ax[1].set_title('Hough Transform^2')

        ax[2].imshow(label_from_plot, cmap='gray')
        ax[2].set_title('Label\'s Hough Transform')

        for a in ax.ravel():
            a.axis('off')

        plt.tight_layout()
        plt.show()

    if display_sift:
        sift = cv2.SIFT()
        kp = sift.detect(image, None)
        temp = cv2.drawKeypoints(image, kp)

        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].imshow(image_from_plot, cmap='gray')
        ax[0].set_title('Original Image')
        
        ax[1].imshow(temp, cmap='gray')
        ax[1].set_title('SURF Features')

        for a in ax.ravel():
            a.axis('off')

        plt.tight_layout()
        plt.show()