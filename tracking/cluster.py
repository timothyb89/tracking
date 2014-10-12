# -*- coding: utf-8 -*-
from point import SimplePoint


class Cluster:

    def __init__(self, points):
        self.points = points


def get_center(points):
    cx = 0.0
    cy = 0.0

    for point in points:
        cx += point.x
        cy += point.y

    return SimplePoint(cx / len(points), cy / len(points))


def find_clusters(points, max_radius = 75, max_length = 0):
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
    cluster_map = {}
    clusters = []
    for a, b, dist_sq, center in distances_sorted:
        if (a in cluster_map) and (b in cluster_map):
            # points have already been paired, move on
            continue
        elif (not a in cluster_map) and (not b in cluster_map):
            # neither point has been added to a cluster
            # because we traverse in descending order of distance, and the
            # distance was already verified, we know we can create a
            # new valid cluster from these points
            cluster = ({a, b}, center)
            cluster_map[a] = cluster
            cluster_map[b] = cluster
            clusters.append(cluster)
        elif (a in cluster_map) and (not b in cluster_map):
            # we want to consider adding b to an existing cluster containing a
            # simulate the addition first to ensure adding this point will not
            # violate our max radius
            cluster, center = cluster_map[a]

            if (max_length > 0) and len(cluster) >= max_length:
                continue

            cluster_sim = list(cluster) + [b]
            center_sim = get_center(cluster_sim)

            valid = True
            for point in cluster_sim:
                if center_sim.distance_squared(point) > max_radius_sq:
                    valid = False
                    break

            if valid:
                # append the point to the cluster and update the center
                center.update(center_sim)
                cluster.add(b)
                cluster_map[b] = (cluster, center)
        elif (not a in cluster_map) and (b in cluster_map):
            # same as above, but we want to add a into b's valid cluster
            cluster, center = cluster_map[b]

            if (max_length > 0) and len(cluster) >= max_length:
                continue

            cluster_sim = list(cluster) + [a]
            center_sim = get_center(cluster_sim)

            valid = True
            for point in cluster_sim:
                if center_sim.distance_squared(point) > max_radius_sq:
                    valid = False
                    break

            if valid:
                center.update(center_sim)
                cluster.add(a)
                cluster_map[a] = (cluster, center)

    return clusters
