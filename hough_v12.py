import os
import cv2
import skimage
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.feature import canny
from sklearn.cluster import KMeans
from matplotlib.figure import Figure
from skimage.draw import circle_perimeter
from sklearn.metrics import silhouette_score
from sklearn.cluster import AffinityPropagation
from sklearn.linear_model import LinearRegression
from skimage.segmentation import flood, flood_fill
from skimage.transform import probabilistic_hough_line
from skimage.transform import hough_circle, hough_circle_peaks
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

    ### Do the circle extraction algorithm
    fig, axes = plt.subplots(1, 3, figsize=(30, 15))
    ax = axes.ravel()

    # Binary threshold
    _, gray = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    
    # Morphological improvements of the mask
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    # Find contours
    cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter large size contours; at the end, there should only be one left
    largeCnts = []
    for cnt in cnts:
        if (cv2.contourArea(cnt) > 10000):
            largeCnts.append(cnt)

    # Draw (filled) contour(s)
    gray = np.uint8(np.zeros(gray.shape))
    gray = cv2.drawContours(gray, largeCnts, -1, 255, cv2.FILLED)    

    # Use hough transform to find circle
    circle = gray.copy()
    circle = cv2.cvtColor(circle, cv2.COLOR_GRAY2RGB)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 40, 2000) #, param1=50, param2=30, minRadius=0, maxRadius=0)

    # If some circle is found
    if circles is not None:
    # Get the (x, y, r) as integers
        circles = np.round(circles[0, :]).astype("int")        
        # loop over the circles
        for (x, y, r) in circles:
            center = [x, y]
            cv2.circle(circle, (x, y), r, (255, 0, 0), 10)            

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_xlim((0, image.shape[0]))
    ax[0].set_ylim((0, image.shape[1]))

    ax[1].imshow(circle, cmap=cm.gray)
    ax[1].plot(center[0], center[1], 'ro', markersize=12)
    ax[1].set_xlim((0, circle.shape[0]))
    ax[1].set_ylim((0, circle.shape[1]))

    ax[2].imshow(image, cmap=cm.gray)
    ax[2].plot(center[0], center[1], 'ro', markersize=12)
    ax[2].set_xlim((0, image.shape[0]))
    ax[2].set_ylim((0, image.shape[1]))

    colors = ['green', 'royalblue', 'orange', 'firebrick', 'teal', 'indigo', 'deeppink', 'lawngreen', 'sienna', 'olive', 'tan', 'coral', 'rosybrown', 'mediumpurple']
    flattened_lines = []
    for lines in extracted_lines:
        
        color = colors[0]
        del colors[0]

        for line in lines:
            flattened_lines.append(line)

            m, b = line
            x_vals = np.array(ax[1].get_xlim())
            y_vals = b + m * x_vals              
            ax[2].plot(x_vals, y_vals, c=color)   

    intersections = []
    for line in flattened_lines:
        for other_line in flattened_lines:
            if line == other_line:
                continue

            m1, b1 = line
            m2, b2 = other_line

            xi = (b1-b2) / (m2-m1)
            yi = m1 * xi + b1

            if xi > 1E308 or yi > 1E308 or xi < -1E308 or yi < -1E308:
                continue

            if xi > image.shape[0] or xi < 0 or yi > image.shape[1] or yi < 0:
                continue
            
            intersections.append([xi, yi])
            ax[2].plot(xi, yi, 'bo', markersize=3)

    # Find mean of intersections
    intersections = np.array(intersections)   
    x_mean = np.mean(intersections[:, 0])
    y_mean = np.mean(intersections[:, 1])         

    ax[2].plot(x_mean, y_mean, 'go', markersize=12)    

    with open(path + '/vector.txt', 'w') as f:
        # Calculate vector
        x_diff = x_mean - center[0]
        y_diff = y_mean - center[1]
        f.write('Vector: <' + str(x_diff) + ', ' + str(y_diff) + '>\n')

        # Calculate magnitude of vector
        mag = np.sqrt(((x_mean - center[0]) ** 2)) + ((y_mean - center[1]) ** 2)
        f.write('Vector magnitude: ' + str(mag))

    plt.savefig(path + "/flood.png")    
    print("Completed:", i)