import cv2
import numpy as np
import random as rng
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and show it
im = cv2.imread(args["image"])
cv2.imshow("image", im)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 113, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)[-2:]
idx = 0

for cnt in contours:
    idx += 1
    x, y, w, h = cv2.boundingRect(cnt)
    roi = im[y:y + h, x:x + w]
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    #cv2.rectangle(im,(x,y),(x+w,y+h),color,2)
    cv2.drawContours(im, [cnt], 0, color, -1)
cv2.imshow('img', im)
cv2.waitKey(0)
