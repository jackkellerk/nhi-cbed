import os
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.feature import canny
from sklearn.cluster import KMeans
from matplotlib.figure import Figure
from sklearn.metrics import silhouette_score
from sklearn.cluster import AffinityPropagation
from sklearn.linear_model import LinearRegression
from skimage.transform import probabilistic_hough_line
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def extract_lines(image, all_info = False):

    # Get probabilistic hough transform
    edges = canny(image, 2, 1, 25)
    out = probabilistic_hough_line(edges, threshold=250, line_length=10, line_gap=3)

    ###########

    # Get a list of the lines
    lines = []
    for line in out:
        p0, p1 = line
        
        # Extrapolate line
        m = (p0[1] - p1[1]) / (p0[0] - p1[0])
        b = p0[1] - (m * p0[0])

        # Append it to lines output
        lines.append([m, b])
    
    lines = np.array(lines)
    lines = lines[np.argsort(lines[:, 0])]

    ###########

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

    ###########

    clusters = []
    for j in range(optimal_k):        
        if lines[kmeans.labels_ == j].shape[0] > 0:
            clusters.append(lines[kmeans.labels_ == j]) # TODO: Maybe only do clusters that have two distinct clusters and points > 5
               
    ###########
    
    # Find the means of the clusters     

    stds = []
    x_means = []
    y_means = []
    for j in range(len(clusters)):
        stds.append(np.std(lines[kmeans.labels_ == j][:, 1]))
        x_means.append(np.mean(lines[kmeans.labels_ == j][:, 0]))
        y_means.append(np.mean(lines[kmeans.labels_ == j][:, 1]))    

    ###########

    # Return lines

    extracted_lines = []

    for j in range(len(clusters)):
        m = x_means[j]
        b1 = y_means[j] + stds[j] 
        b2 = y_means[j] - stds[j]

        extracted_lines.append([[m, b1], [m, b2]]) # TODO: Here, find the closest points to these coordinates rather than just arbitarily choosing based on STD

    return [extracted_lines, lines, clusters] if all_info else extracted_lines

# TODO: Create a function to determine the movement vector

###########
###########
###########

# Plot lines on graph

for i in range(1, 21):        
    image = cv2.imread("data/train/samples/CBED_" + str(i) + ".tif", cv2.IMREAD_GRAYSCALE)    

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_xlim((0, image.shape[0]))
    ax[0].set_ylim((0, image.shape[1]))

    ax[1].imshow(image, cmap=cm.gray)
    ax[1].set_xlim((0, image.shape[0]))
    ax[1].set_ylim((0, image.shape[1]))    

    ax[2].imshow(image, cmap=cm.gray)
    ax[2].set_xlim((0, image.shape[0]))
    ax[2].set_ylim((0, image.shape[1]))     

    extracted_lines, all_lines, clusters = extract_lines(image, all_info=True)

    for line in all_lines:
        m, b = line
        x_vals = np.array(ax[0].get_xlim())
        y_vals = b + m * x_vals
        ax[1].plot(x_vals, y_vals, alpha=0.5, c='r')        

    colors = ['green', 'royalblue', 'orange', 'firebrick', 'teal', 'indigo', 'deeppink', 'lawngreen', 'sienna', 'olive', 'tan', 'coral', 'rosybrown', 'mediumpurple']
    for lines in extracted_lines:
        
        color = colors[0]
        del colors[0]

        for line in lines:
            m, b = line
            x_vals = np.array(ax[1].get_xlim())
            y_vals = b + m * x_vals
            ax[1].plot(x_vals, y_vals, c='b')    
            ax[2].plot(x_vals, y_vals, c=color)    

    path = "./outputimages/" + str(i)
    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.savefig(path + "/out.png")


    fig, axes = plt.subplots(1, 2, figsize=(30, 15))
    ax = axes.ravel()

    colors = ['green', 'royalblue', 'orange', 'firebrick', 'teal', 'mediumpurple', 'deeppink', 'lawngreen', 'sienna', 'olive', 'tan', 'coral', 'rosybrown', 'indigo']
    for j in range(len(clusters)):

        color = colors[0]
        del colors[0]

        ax[0].scatter(clusters[j][:, 0], [0 for k in range(len(clusters[j][:, 0]))], c=color)
        ax[1].scatter(clusters[j][:, 0], clusters[j][:, 1], c=color)
    
    plt.savefig(path + "/data.png")
    print("Completed:", i)

# TODO: only select the clusters with low entropy and draw those. also, replace the right subgraph with a graph with all three (the image, the lines, the extracted lines)
# TODO: look into why 11, 12, 16, 17, 22 look wierd. Some line clusters do not look like they correspond to anything on the left? incorrect math? Answer below:
# Just did some research. yes, 11 looks like a OLS looks like a terrible fit. Just did mean of clsuter instead, probs better.