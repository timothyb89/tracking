# -*- coding: utf-8 -*-
"""
General color-related functionality, in particular color approximation..
"""

import numpy as np

# color definitions, (h,s,v) from 0..1
COLORS = {
    'blue':   (1.0, 0.0, 0.0),
    'green':  (0.40, 1.00, 0.30),
    'yellow': (0.00, 1.00, 0.60),
    'red':    (0.15, 0.30, 0.60),
    'pink':   (1.00, 0.50, 0.60),
    'white':  (0.50, 0.10, 0.50)
}


pi2 = np.pi * 2
pi1_3 = np.pi / 3


def bgr_to_hsv((b, g, r), convert = True):
    if convert:
        b /= 255.0
        g /= 255.0
        r /= 255.0

    min_ = min(r, g, b)
    max_ = max(r, g, b)
    range_ = max_ - min_

    if max_ == 0 or range_ == 0:
        return 0, 0, 0

    # H'
    if r == max_:
        hue = (g - b) / range_
    elif g == max_:
        hue = 2 + (b - r) / range_
    else: # b == max
        hue = 4 + (r - g) / range_

    # H' -> H
    hue *= pi1_3
    if hue < 0:
        hue += pi2

    # 0.0 - 1.0 (mapped to 0 - 2pi)
    hue_ratio = hue / pi2

    saturation = range_ / max_
    value = max_

    return hue_ratio, saturation, value


def distance_squared(color_a, color_b):
    """
    Simple Euclidean distance for floating point colors.

    :param color_a: a 3-tuple
    :param color_b: a 3-tuple
    :return: the squared distance between the two color points
    """
    ay, au, av = color_a
    by, bu, bv = color_b

    return (by - ay)**2 + (bu - au)**2 + (bv - av)**2


def get_color(color_hsv, tolerance = 0.5):
    distances = { name: distance_squared(c, color_hsv) for name, c in COLORS.iteritems() }

    colors_sorted = sorted(
        COLORS.keys(),
        key = lambda k: distances[k])

    if tolerance > 0:
        colors_sorted = filter(lambda color: distances[color] <= tolerance, colors_sorted)

    return colors_sorted, distances
