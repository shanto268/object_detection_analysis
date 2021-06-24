from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2
from sklearn import cluster

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-n",
                "--number",
                required=True,
                type=int,
                help="Number of Colors")
args = vars(ap.parse_args())
number = args["number"]

# load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("image", image)

# convert the image to grayscale and create a histogram
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

# grab the image channels, initialize the tuple of colors,
# the figure and the flattened feature vector
chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

features = []
# loop over the image channels
for (chan, color) in zip(chans, colors):
    # create a histogram for the current channel and
    # concatenate the resulting histograms for each
    # channel
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    features.extend(hist)
    # plot the histogram
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.show()

# here we are simply showing the dimensionality of the
# flattened color histogram 256 bins for each channel
# x 3 channels = 768 total values -- in practice, we would
# normally not use 256 bins for each channel, a choice
# between 32-96 bins are normally used, but this tends
# to be application dependent

# read image into range 0 to 1
img = image / 255

# set number of colors

# quantize to 16 colors using kmeans
h, w, c = img.shape
img2 = img.reshape(h * w, c)
kmeans_cluster = cluster.KMeans(n_clusters=number)
kmeans_cluster.fit(img2)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_labels = kmeans_cluster.labels_

# need to scale back to range 0-255 and reshape
img3 = cluster_centers[cluster_labels].reshape(h, w, c) * 255.0
img3 = img3.astype('uint8')

cv2.imshow('reduced colors', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# reshape img to 1 column of 3 colors
# -1 means figure out how big it needs to be for that dimension
img4 = img3.reshape(-1, 3)

# get the unique colors
colors, counts = np.unique(img4, return_counts=True, axis=0)

# compute HSV Value equals max(r,g,b)
values = []
for color in colors:
    b = color[0]
    g = color[1]
    r = color[2]
    v = max(b, g, r)
    values.append(v)

# zip colors, counts, values together
unique = zip(colors, counts, values)

# make list of color, count, value
ccv_list = []
for color, count, value in unique:
    ccv_list.append((color, count, value))


# function to define key as third element
def takeThird(elem):
    return elem[2]


# sort ccv_list by Value (brightness)
ccv_list.sort(key=takeThird)

# plot each color sorted by increasing Value (brightness)
# pyplot uses normalized r,g,b in range 0 to 1
fig = plt.figure()
length = len(ccv_list)
for i in range(length):
    item = ccv_list[i]
    color = item[0]
    b = color[0] / 255
    g = color[1] / 255
    r = color[2] / 255
    count = item[1]
    plt.bar(i, count, color=((r, g, b)))

# show and save plot
plt.show()
#fig.savefig('barn_color_histogram2.png')
plt.close(fig)
