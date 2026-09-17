import cv2 as cv
import numpy as np


def create_mask(hsv, lower, upper):
    return cv.inRange(hsv, lower, upper)


def find_contours(mask):
    contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    return contours


def get_largest_contours(contours, n=2, min_area=1000):

    contours = [
        c for c in contours
        if cv.contourArea(c) >= min_area
    ]

    contours = sorted(
        contours,
        key=cv.contourArea,
        reverse=True
    )

    return contours[:n]


def approximate_contour(contour, epsilon_ratio=0.02):

    epsilon = epsilon_ratio * cv.arcLength(
        contour,
        True
    )

    approx = cv.approxPolyDP(
        contour,
        epsilon,
        True
    )

    return approx


def get_y_range(contours):

    all_points = []

    for contour in contours:

        approx = approximate_contour(contour)

        points = approx.reshape(-1, 2)

        all_points.extend(points)

    all_points = np.array(all_points)

    y_min = np.min(all_points[:, 1])
    y_max = np.max(all_points[:, 1])

    return y_min, y_max


def calculate_distances(cy, y_min, y_max):

    d_top = abs(cy - y_min)
    d_bottom = abs(cy - y_max)

    return d_top, d_bottom