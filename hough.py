import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks

# Open image and labels
image = cv2.imread("data/train/samples/CBED_1.tif", cv2.IMREAD_GRAYSCALE)
label = cv2.imread("data/train/labels/CBED_1.png")

# Classic straight-line Hough transform
# Set a precision of 0.5 degree.
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
h, theta, d = hough_line(image, theta=tested_angles)

# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(label[:,:,::-1])
ax[1].set_title('Input image labels')
ax[1].set_axis_off()

angle_step = 0.5 * np.diff(theta).mean()
d_step = 0.5 * np.diff(d).mean()
bounds = [np.rad2deg(theta[0] - angle_step),
          np.rad2deg(theta[-1] + angle_step),
          d[-1] + d_step, d[0] - d_step]

'''
ax[3].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect= 1 / 1.5)
ax[3].set_title('Hough transform')
ax[3].set_xlabel('Angles (degrees)')
ax[3].set_ylabel('Distance (pixels)')
ax[3].axis('image')
'''

ax[2].imshow(image, cmap=cm.gray)
ax[2].set_xlim((image.shape[0], 0))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

s = 0
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):        
    s += 1
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])    
    ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))

print(s)
plt.tight_layout()
plt.show()