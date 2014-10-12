# -*- coding: utf-8 -*-
"""
Functionality and tunables for interpretation of raw camera images.

Fundamentally, this module takes and input image, and processes it to eventually extract a number of candidate
Circle instances.
"""

import cv2
import numpy as np

from point import Point, SimplePoint
from color import bgr_to_hsv

kernel = np.ones((3, 3), np.uint8)
pi4 = np.pi * 4


class Circle:
    """
    A raw candidate point. These can be passed to the point tracking algorithm :mod:`tracking.point`.
    """

    def __init__(self, frame, contour, color, x, y, radius, circularity):
        self.frame = frame
        self.contour = contour
        self.color = color
        self.x = x
        self.y = y
        self.radius = radius
        self.circularity = circularity

    @property
    def pos(self):
        return int(self.x), int(self.y)


def preprocess(frame):
    """
    Preprocesses the given frame. Currently this only applies a Gaussian blur to eliminate some noise.

    :param frame: the frame to process
    :return: the processed frame
    """
    return cv2.GaussianBlur(frame, (5, 5), 2)


def find_edges(frame):
    """
    Finds edges in a raw or preprocessed color image. A mask will be applied to
    filter for only (largely) dark areas.

    :param frame: the raw or preprocessed BGR frame
    :return: the frame with Canny edge detection applied to regions of interest
    """

    # find black areas + erode, dilate to eliminate dots
    eroded = cv2.erode(frame, kernel, iterations = 15)

    # erosion here would reduce a lot of invalid search area, but it's
    # expensive and the point tracking is robust enough that it isn't necessary
    #eroded = cv2.dilate(eroded, kernel, iterations = 20)

    ret, black = cv2.threshold(eroded, 40, 255, cv2.THRESH_BINARY_INV)

    # merge into single binary image to use as mask
    bb, bg, br = cv2.split(black)
    all_black = cv2.bitwise_and(br, bg)
    all_black = cv2.bitwise_and(all_black, bb)
    search_area = cv2.bitwise_and(frame, frame, mask = all_black)

    # convert to grayscale to use for edge detection
    black_region = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)

    return cv2.Canny(black_region, 100, 50)


def find_circles(frame, frame_count, edges):
    """
    Given an edge-detected frame, locates contour candidates and returns a list
    of Circle instances.

    :param frame: the original (or preprocessed) frame
    :param frame_count: the current frame number
    :param edges: the edge detected
    :return: a list of located Circle instances.
    """
    circles = []

    cimg, contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 30 or area > 700:
            continue

        _, (w, h), angle = cv2.minAreaRect(contour)
        if w < 1 or h < 1:
            continue

        ratio = w / h
        if ratio < 0.25 or ratio > 1.75:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > 25 or radius < 1.5:
            continue

        arclen = cv2.arcLength(contour, True)
        circularity = (pi4 * area) / (arclen * arclen)
        if circularity < 0.60:
            continue

        # create a mask for the contour area
        height, width, _ = frame.shape
        mask = np.zeros((height, width, 1), np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)

        # find the mean color within the mask
        b, g, r, _ = cv2.mean(frame, mask = mask)
        hsv = bgr_to_hsv((b, g, r))

        circles.append(Circle(frame_count, contour, hsv, x, y, radius, circularity))

    return circles

