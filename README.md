# Kikuchi Line Detection from CBED images
Jack Kellerk in coordination with the Nano Human Interface (NHI) Research Lab at Lehigh University

## ./data
This is the data folder

## ./output
This is the output folder for hough_v6.py (below)

## ./line-detection-using-cnns
This is the simple neural network that I used on the CBED images. It has its own README.md

## hough.py
This is the standard hough transform on the CBED images

## hough_v2.py
This is the probabilistic hough transform on the CBED images using canny edge detection preprocessing

## hough_v3.py
This is the quantitative approach of hough_v2.py on every image in the dataset using the accuracy and RMSE metric I developed

## hough_v4.py
This is extrapolating every line segment across the entire image from hough_v2.py

## hough_v5.py
This is using hough_v2.py but plotting each line segment in cartesian space and using K-means to cluster pockets of line segments

## hough_v6.py
This is saving the probabilistic hough transform results for each image in the dataset for determining how well this method works qualitatively across all images

## Misc. python files
These are used when developing the primary methods above. They do not contain important code/results
