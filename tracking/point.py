# -*- coding: utf-8 -*-
"""
Functionality related to point tracking.

This module can accept a list of candidate points (Circle instances) and output a qualitatively more "stable" list
of Point instances.
"""
import cv2

import math
import numpy as np
import numpy.linalg as la

from collections import deque

from color import get_color

#
# Tunables
#

#: Total statistic history length to maintain, in frames
BUFFER_LENGTH = 50

#: The maximum distance, in pixels, a point can travel
MAX_DISTANCE = 320

#: The "window size" used for running averages
WINDOW_SIZE = 10

#: The minimum point health value before points are removed
FRAME_TIMEOUT = 20

#: The maximum point health
POINT_MAX_HEALTH = 50

#: The minimum velocity to use during predictions, in pixels-per-frame
VELOCITY_PREDICT_MINIMUM = 1.0

#: A multiplier on the point search bounds.
#: Additionally used as the offset for
BOUNDS_MULTIPLIER = 20

#: An angle of rotation for search area upper and lower bounds.
ROTATION_THETA = np.pi / 6

#
# End of tunables
#

ROTATION_CCW = np.mat([
    [np.cos(ROTATION_THETA), -np.sin(ROTATION_THETA)],
    [np.sin(ROTATION_THETA),  np.cos(ROTATION_THETA)]
])

ROTATION_CW = np.mat([
    [np.cos(-ROTATION_THETA), -np.sin(-ROTATION_THETA)],
    [np.sin(-ROTATION_THETA),  np.cos(-ROTATION_THETA)]
])

point_index = 0


class SimplePoint:
    """
    A simple point class with only basic functionality for position, distance, and midpoint calculation.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def pos(self):
        return int(self.x), int(self.y)

    def distance_squared(self, other):
        return (other.x - self.x)**2 + (other.y - self.y)**2

    def midpoint(self, other):
        cx = (self.x + other.x) / 2.0
        cy = (self.y + other.y) / 2.0

        return SimplePoint(cx, cy)

    def update(self, other):
        self.x = other.x
        self.y = other.y


class Point:
    """
    A Point class containing tracking data and functionality for a particular known point.
    """

    def __init__(self, circle):
        global point_index
        self.index = point_index
        point_index += 1

        self.frame_numbers = deque(maxlen = WINDOW_SIZE)
        self.circle_history = deque(maxlen = WINDOW_SIZE)

        self.health = 0

        self.x_window = deque(maxlen = WINDOW_SIZE)
        self.x_window_mean = 0
        self.x_velocity_window = []
        self.x_velocity_mean = 0

        self.y_window = deque(maxlen = WINDOW_SIZE)
        self.y_window_mean = 0
        self.y_velocity_window = []
        self.y_velocity_mean = 0

        self.circularity_window = deque(maxlen = WINDOW_SIZE)
        self.circularity_mean = 0

        self.color_window = deque(maxlen = WINDOW_SIZE)
        self.color_mean = (0, 0, 0)

        self.search_bounds = None

        self.update(circle)

    def update(self, circle):
        self.frame_numbers.append(circle.frame)
        self.circle_history.append(circle)

        self.x_window.append(circle.x)
        self.y_window.append(circle.y)
        self.circularity_window.append(circle.circularity)
        self.color_window.append(circle.color)

        self.x_window_mean = np.mean(self.x_window)
        self.y_window_mean = np.mean(self.y_window)
        self.circularity_mean = np.mean(self.circularity_window)
        self.color_mean = np.mean(self.color_window, axis = 0)

        self.x_velocity_window = np.diff(self.x_window)
        self.y_velocity_window = np.diff(self.y_window)

        self.x_velocity_mean = np.mean(self.x_velocity_window)
        self.y_velocity_mean = np.mean(self.y_velocity_window)

        if len(self.x_window) >= 2:
            self.search_bounds = self.get_search_bounds()

        if self.health < POINT_MAX_HEALTH:
            self.health += 1

    def update_empty(self, frame):
        """
        Updates this point for an "empty" frame where no matching circle was located.

        :param frame: the current frame number
        """
        # decay is "stronger" than growth (-5  instead of -1)
        self.health -= 5

    def is_expired(self, frame):
        return self.health < -FRAME_TIMEOUT

    def predicted_linear_distance(self, circle):
        """
        Forms a line based on the current known direction of this point, and
        finds the distance between the line and the given circle or point-like
        object.

        :param circle: a Circle or Point-like object with an `x` and a `y`
        :return: the predicted distance, or None
        """
        if len(self.x_window) < 2:
            return None

        # line in vector form, x = a + tn
        a = np.array([ [self.x], [self.y] ], dtype = float)

        n = np.array([ [self.x_velocity_mean], [self.y_velocity_mean] ])
        n /= la.norm(n)  # convert to unit vector

        # point of interest
        p = np.array([ [circle.x], [circle.y] ])

        return la.norm((a - p) - ((a - p) * n) * n)

    def predicted_pos(self, current_frame = -1):
        """
        Determines a predicted position for the frame `current_frame` frames into the future.

        :param current_frame: the current frame number
        :return: a predicted (x, y) tuple
        """
        if len(self.x_window) < 2:
            # can't generate a prediction, so return the current location
            # (we'll just do a minimum distance against all points for the first 2 frames)
            return self.pos

        # TODO: the multiplier currently only assumes some linear velocity
        # TODO: we could factor in acceleration to get a potentially more accurate prediction

        # attempt to account for travel over multiple frames based on velocity
        # if no frame is specified, assume
        if current_frame == -1:
            multiplier = 1
        else:
            multiplier = current_frame - self.last_frame

        # TODO: velocity mean vs last?
        px = self.x_window[-1] + (self.x_velocity_mean * multiplier)
        py = self.y_window[-1] + (self.y_velocity_mean * multiplier)

        return px, py

    def get_search_bounds(self):
        """
        Generates a polygon representing the search bounds for matching candidate points. Requires a window history of
        at least 2 previous points.

        :return: a list of (x, y) tuples
        """
        if len(self.x_window) < 2:
            return None

        # form a triangle bounding a predicted search area
        # the 'height' of the triangle is the extrapolated position times some constant
        # the two base points are the 'height' line rotated by some constant

        # line in vector form, x = a + tn

        # a: the current position vector of the point
        a = np.array([[self.x], [self.y]], dtype=float)

        # the direction vector, it's norm, and the equivalent unit vector
        n = np.array([[self.x_velocity_mean], [self.y_velocity_mean]])
        n_norm = la.norm(n)
        n_uv = n / n_norm

        # move the top of the triangle backward some to allow for room for backwards movement
        rx, ry = a - BOUNDS_MULTIPLIER * n_uv

        # counter-clockwise rotated point
        uv_ccw = ROTATION_CCW * n_uv
        ccw_x, ccw_y = a + BOUNDS_MULTIPLIER * n_norm * uv_ccw

        # clockwise rotated point
        uv_cw = ROTATION_CW * n_uv
        cw_x, cw_y = a + BOUNDS_MULTIPLIER * n_norm * uv_cw

        # express the points as an opencv contour
        return np.array([
            [rx, ry],
            [ccw_x, ccw_y],
            [cw_x, cw_y]
        ], dtype=np.float32)

    def in_predicted_bounds(self, circle):
        """
        Determines if the given point-like object is within the predicted bounds for this class.

        :param circle: the point-like object to examine
        :return: True if the object is inside the bounds of the prediction, False otherwise
        """

        if len(self.x_window) < 2:
            return True

        # TODO: look into replacing with a simplified within-triangle test
        # this is a fairly expensive method call, a more efficient method probably exists

        return cv2.pointPolygonTest(self.search_bounds, (circle.x, circle.y), False)

    def predicted_distance_squared(self, circle):
        px, py = self.predicted_pos(circle.frame)

        return (circle.x - px)**2 + (circle.y - py)**2

    def distance_squared(self, point):
        """
        Returns the Euclidean distance between this point and the center of the
        given point-like object..

        :param point: the point to compare against
        :return: the squared Euclidean distance between the two points
        """
        return (point.x - self.x_window_mean)**2 + (point.y - self.y_window_mean)**2

    def midpoint(self, other):
        """
        Determines the midpoint between this point and some other point-like object.

        :param other: a point-like object
        :return: a :class:`SimplePoint` instance at the position of the midpoint
        """

        cx = (self.x + other.x) / 2.0
        cy = (self.y + other.y) / 2.0

        return SimplePoint(cx, cy)

    def draw(self, image, frame):
        """
        Draws a representation of this point onto the given image.

        :param image: the OpenCV mat to draw onto
        """

        if self.quality > 0.25:
            #if len(self.x_window) >= 2:
            #    cv2.ellipse(image, (self.pos, self.search_bounds, 0), (0, 255, 255))

            #cv2.circle(image, self.pos, 3, (255, 255, 255), -1)

            px, py = self.predicted_pos(frame)
            ppos = (int(px), int(py))
            cv2.line(image, self.pos, ppos, (255, 255, 255), 1)
            #cv2.circle(image, ppos, 2, (0, 0, 255), -1)

            #cv2.putText(image, "%d - %.2f - %s" % (self.index, self.quality, self.color),
            #            (self.x + 5, self.y + 10),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), lineType = cv2.LINE_AA)

            cv2.putText(image, "%d" % self.index,
                        (self.x + 5, self.y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), lineType = cv2.LINE_AA)

            #m = self.slope_mean
            #if m is not None:
            #    x, y = self.pos
            #    cv2.line(image, (x, y), (x + 20, int(y + 20 * m)), (0, 255, 0))
            #    cv2.line(image, (x, y), (x - 20, int(y - 20 * m)), (0, 255, 0))

            if not np.isnan(self.x_velocity_mean) and not np.isnan(self.y_velocity_mean):
                # draw extrapolated line (zero-safe)
                x, y = self.pos
                cv2.line(image, self.pos,
                         (int(x + 20 * self.x_velocity_mean), int(y + 20 * self.y_velocity_mean)),
                         (0, 255, 0))


                # if velocity is nonzero, draw the bounding rect
                if self.x_velocity_mean != 0 or self.y_velocity_mean != 0:
                    # line in vector form, x = a + tn
                    a = np.array([ [self.x], [self.y] ], dtype = float)

                    n = np.array([ [self.x_velocity_mean], [self.y_velocity_mean] ])
                    n_norm = la.norm(n)
                    uv = n / n_norm  # convert to unit vector

                    rx, ry = a - 20 * uv

                    uv_ccw = ROTATION_CCW * uv
                    ccw_x, ccw_y = a + 20 * n_norm * uv_ccw
                    cv2.line(image, (int(rx), int(ry)), (int(ccw_x), int(ccw_y)), (0, 255, 0))

                    uv_cw = ROTATION_CW * uv
                    cw_x, cw_y = a + 20 * n_norm * uv_cw
                    cv2.line(image, (int(rx), int(ry)), (int(cw_x), int(cw_y)), (0, 255, 0))



            #if len(self.circle_history) >= 1:
            #    c = self.circle_history[-1]

            #    cv2.circle(image, c.pos, 1, (0, 255, 255), 1)
            #    cv2.putText(image, "%.2f" % self.predicted_linear_distance(c),
            #            (int(c.x + 5), int(c.y + 10)),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), lineType = cv2.LINE_AA)

            #h, s, v = self.color_mean
            #cv2.putText(image, "%.2f %.2f %.2f" % (h, s, v),
            #            (self.x, self.y + 20),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), lineType=cv2.LINE_AA)


    @property
    def x(self):
        return int(self.x_window_mean)

    @property
    def y(self):
        return int(self.y_window_mean)

    @property
    def last_x(self):
        return int(self.x_window[-1])

    @property
    def last_y(self):
        return int(self.y_window[-1])

    @property
    def pos(self):
        #return self.x_history[-1], self.y_history[-1]
        return int(self.x_window_mean), int(self.y_window_mean)

    @property
    def last_frame(self):
        return self.frame_numbers[-1]

    @property
    def quality(self):
        health = float(self.health)
        if health < 0:
            health = 0.0

        return (0.75 * (health / POINT_MAX_HEALTH)) + (0.25 * self.circularity_mean)

    @property
    def color(self):
        colors, distances = get_color(self.color_mean)

        if colors:
            return colors[0]
        else:
            return None


def find_points(circles, points, frame_count):
    """
    Given a list of Circle instances, creates or updates Point instances. The
    passed list of known Point objects will be modified.

    :param circles: a list of circles
    :param points: a list of previously known points
    :param frame_count: the current frame number
    :return:
    """
    # attempt to pair points with a globally minimum-distance contour
    # we need to gather all valid pairs and attempt to minimize distances for all of them
    expired = []
    distances = []
    for point in points:
        if point.is_expired(frame_count):
            expired.append(point)
            continue

        for circle in circles:
            if not point.in_predicted_bounds(circle):
                continue

            dist = point.predicted_distance_squared(circle)

            if dist < MAX_DISTANCE:
                distances.append((point, circle, dist))

    for point in expired:
        points.remove(point)

    paired_circles = []
    for (point, circle, distance) in sorted(distances, key=lambda d: d[2]):
        if circle in paired_circles:
            continue

        point.update(circle)
        paired_circles.append(circle)

    # the remaining circles are previously unknown
    remaining_circles = [c for c in circles if c not in paired_circles]
    for circle in remaining_circles:
        p = Point(circle)
        points.append(p)

    # iterate again to find all remaining points and "empty" update them
    for point in points:
        if point.last_frame != frame_count:
            point.update_empty(frame_count)

    return points


def get_center(points):
    cx = 0.0
    cy = 0.0

    for point in points:
        cx += point.x
        cy += point.y

    return SimplePoint(cx / len(points), cy / len(points))
