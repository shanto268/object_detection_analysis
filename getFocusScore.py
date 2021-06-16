# importing required libraries of opencv
import cv2, glob
from matplotlib import pyplot as plt

im_dir = glob.glob(
    "/Users/sshanto/hep/hep_daq/CAMAC/object_detection_analysis/pixel_tomograms/multi/*"
)


def getMean(hist):
    score = 0
    for i in range(len(hist)):
        score += hist[i] * i
    return score[0]


def getFocus(im_dir):
    fm = []
    fms = []
    z_dist = []

    for image in im_dir:
        z = int(image.split("/")[-1].split(".")[0].split('img')[-1])
        # reads an input image
        img = cv2.imread(image, 0)
        # find frequency of pixels in range 0-255
        histr = cv2.calcHist([img], [0], None, [256], [0, 256])
        fms.append(histr)
        fm.append(getMean(histr))
        z_dist.append(z)

    # show the plotting graph of an image
    coord = zip(z_dist, fm)
    coord = sorted(list(coord), key=lambda t: t[0])
    z, score = list(zip(*coord))
    plt.plot(z, score, '--o')
    plt.show()
