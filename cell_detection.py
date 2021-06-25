import cv2
import numpy as np
import math
import argparse


def analyzeTomogram(img):
    image = cv2.imread(img)
    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hsv_lower = np.array([170, 201, 81])
    hsv_upper = np.array([194, 156, 68])  #161,0,0
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    minimum_area = 20
    average_cell_area = 65
    connected_cell_area = 100
    cells = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > minimum_area:
            cv2.drawContours(original, [c], -1, (36, 255, 12), 2)
            if area > connected_cell_area:
                cells += math.ceil(area / average_cell_area)
            else:
                cells += 1
    print('Cells: {}'.format(cells))
    cv2.imshow('Cells Detected: {}'.format(cells), original)
    cv2.imshow("hsv", hsv)
    cv2.imshow('morphological', close)
    cv2.imshow('original', image)
    cv2.waitKey()


def analyzeCells(img):
    image = cv2.imread(img)
    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array([156, 60, 0])
    hsv_upper = np.array([179, 115, 255])
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    minimum_area = 500
    average_cell_area = 1000
    connected_cell_area = 2000
    cells = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > minimum_area:
            cv2.drawContours(original, [c], -1, (36, 255, 12), 2)
            if area > connected_cell_area:
                cells += math.ceil(area / average_cell_area)
            else:
                cells += 1
    print('Objects: {}'.format(cells))
    cv2.imshow('Cells Detected: {}'.format(cells), original)
    # cv2.imshow('morphological', close)
    # cv2.imshow("hsv", hsv)
    # cv2.imshow('original', image)
    cv2.waitKey()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i",
                    "--image",
                    required=True,
                    help="path to the input image")
    ap.add_argument("-t",
                    "--type",
                    default="cell",
                    required=False,
                    help="type of algo")

    args = vars(ap.parse_args())
    img = args["image"]
    typ = args["type"]

    if typ == "tomo":
        analyzeTomogram(img)
    else:
        analyzeCells(img)
