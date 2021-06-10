# -*- coding: utf-8 -*-
"""
================================================
Program : CrateAnalysis/contrast_measure.py
================================================
Summary:

Usage: python3 contrast_measure.py --input examples
"""
__author__ = "Sadman Ahmed Shanto"
__date__ = "06/10/2021"
__email__ = "sadman-ahmed.shanto@ttu.edu"

# import the necessary packages
from skimage.exposure import is_low_contrast
from imutils.paths import list_images
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i",
                "--input",
                required=True,
                help="path to input directory of images")
ap.add_argument("-t",
                "--thresh",
                type=float,
                default=0.35,
                help="threshold for low contrast")
args = vars(ap.parse_args())

# grab the paths to the input images
imagePaths = sorted(list(list_images(args["input"])))

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # load the input image from disk, resize it, and convert it to
    # grayscale
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=450)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur the image slightly and perform edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    # initialize the text and color to indicate that the input image
    # is *not* low contrast
    # text = "Low contrast: No"
    text = ""
    color = (0, 255, 0)

    # check to see if the image is low contrast
    if is_low_contrast(gray, fraction_threshold=args["thresh"]):
        # update the text and color
        # text = "Low contrast: Yes"
        text = ""
        color = (0, 0, 255)
    # otherwise, the image is *not* low contrast, so we can continue
    # processing it
    else:
        # find contours in the edge map and find the largest one,
        # which we'll assume is the outline of our color correction
        # card
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw the largest contour on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    # draw the text on the output image
    cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    # show the output image and edge map
    cv2.imshow("Image", image)
    cv2.imshow("Edge", edged)
    cv2.waitKey(0)
