# -*- coding: utf-8 -*-
import cv2
import numpy as np

from point import SimplePoint

DISTANCE_MULTIPLIER = 35.0
DISTANCE_PHYSICAL   = 12.0


class Cluster:

    def __init__(self, points):
        self.points = set(points)

        self.center = None

        self.update()

    def contains(self, point):
        return point in self.points

    def add(self, point):
        self.points.add(point)

        self.update()

    def remove(self, point):
        self.points.remove(point)

        self.update()

    def update(self):
        if self.is_empty:
            self.center = SimplePoint(0, 0)
        else:
            self.center = get_center(self.points)

    def sim_center(self, point_candidate):
        return get_center(list(self.points) + [point_candidate])

    def predicted_points(self, frame = -1):
        for point in self.points:
            px, py = point.predicted_pos(frame)
            yield SimplePoint(px, py)

    def clean(self, frame):
        to_remove = []

        for point in self.points:
            if point.is_expired(frame):
                to_remove.append(point)

        for point in to_remove:
            self.points.remove(point)

        self.update()

    @property
    def size(self):
        return len(self.points)

    @property
    def is_empty(self):
        return len(self.points) == 0

    @property
    def is_dead(self):
        return len(self.points) < 2

    def sum_of_distances(self):
        sum = 0

        for point in self.points:
            sum += np.sqrt(point.distance_squared(self.center))

        return sum

    def mean_distance(self):
        return self.sum_of_distances() / len(self.points)

    def distance(self):
        return (DISTANCE_MULTIPLIER / self.mean_distance()) * DISTANCE_PHYSICAL

    def draw(self, image):
        ppoints = list(self.predicted_points())
        pcenter = get_center(ppoints)

        for point in ppoints:
            cv2.line(image, point.pos, pcenter.pos, (0, 255, 0), 1)

        cv2.circle(image, pcenter.pos, 4, (255, 255, 255), 2)

        # pairs of point (i, i + 1)
        # point n will be point (i, 0)
        for a, b in zip(ppoints, ppoints[-1:] + ppoints[:-1]):
            cv2.line(image, a.pos, b.pos, (0, 0, 255), 1)

        cv2.putText(image, "%.2f" % (self.distance()),
                    (int(self.center.x) + 5, int(self.center.y + 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), lineType=cv2.LINE_AA)


def get_center(points):
    cx = 0.0
    cy = 0.0

    for point in points:
        cx += point.x
        cy += point.y

    return SimplePoint(cx / len(points), cy / len(points))


def get_cluster(point, clusters):
    for cluster in clusters:
        if cluster.contains(point):
            return cluster

    return None


def find_clusters(points, clusters = None, max_radius = 75, max_length = 0):
    max_radius_sq = max_radius**2
    max_radius_2sq = (max_radius * 2)**2

    # minimum distance pairs
    distances_sq = []
    for a in points:
        for b in points:
            if a is b:
                continue

            # points are too far apart, move on
            # note: we look for distance from centroid, so for these point
            # pairs we can allow checks against the diameter rather than
            # radius
            dist_sq = a.distance_squared(b)
            if dist_sq > max_radius_2sq:
                continue

            distances_sq.append((a, b, dist_sq, a.midpoint(b)))

    distances_sorted = sorted(distances_sq, key = lambda d: d[2])

    # do a final pass to find minimum distance from centroid
    if clusters is None:
        clusters = []

    clusters_remaining = set(clusters)

    for a, b, dist_sq, center in distances_sorted:
        a_cluster = get_cluster(a, clusters)
        b_cluster = get_cluster(b, clusters)

        if a_cluster and b_cluster:
            # points have already been paired, move on
            continue
        elif not a_cluster and not b_cluster:
            # neither point has been added to a cluster
            # because we traverse in descending order of distance, and the
            # distance was already verified, we know we can create a
            # new valid cluster from these points
            cluster = Cluster([a, b])
            clusters.append(cluster)
        elif a_cluster and not b_cluster:
            # we want to consider adding b to an existing cluster containing a
            # simulate the addition first to ensure adding this point will not
            # violate our max radius
            if (max_length > 0) and a_cluster.size >= max_length:
                continue

            center_sim = a_cluster.sim_center(b)

            valid = True
            for point in list(a_cluster.points) + [center_sim]:
                if center_sim.distance_squared(point) > max_radius_sq:
                    valid = False
                    break

            if valid:
                # append the point to the cluster and update the center
                a_cluster.add(b)

                if a_cluster in clusters_remaining:
                    clusters_remaining.remove(a_cluster)
        elif not a_cluster and b_cluster:
            # same as above, but we want to add a into b's valid cluster
            if (max_length > 0) and b_cluster.size >= max_length:
                continue

            center_sim = b_cluster.sim_center(a)

            valid = True
            for point in list(b_cluster.points) + [center_sim]:
                if center_sim.distance_squared(point) > max_radius_sq:
                    valid = False
                    break

            if valid:
                b_cluster.add(a)

                if b_cluster in clusters_remaining:
                    clusters_remaining.remove(b_cluster)

    for cluster in clusters_remaining:
        cluster.update()

    return clusters
