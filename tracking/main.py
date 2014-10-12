import math
import cv2
import numpy as np
import numpy.linalg as la

from collections import deque, namedtuple
from Queue import Queue
from threading import Thread, Event

SIZE = (640, 480)

BUFFER_LENGTH = 50
MAX_DISTANCE = 320
WINDOW_SIZE = 10
FRAME_TIMEOUT = 20
POINT_MAX_HEALTH = 50

from vision import preprocess, find_edges, find_circles
from point import find_points
from cluster import find_clusters


class TrackingThread(Thread):

    def __init__(self, camera_id, name):
        super(TrackingThread, self).__init__()

        self.frames = Queue(maxsize = 1)

        self.capture = cv2.VideoCapture(camera_id)
        self.name = name

        self.frame_count = 0
        self.running = False

        self.points = []
        self.clusters = []

    def process(self, frame):
        frame = preprocess(frame)

        edges = find_edges(frame)
        circles = find_circles(frame, self.frame_count, edges)

        points = find_points(circles, self.points, self.frame_count)
        self.points = points

        acceptable = filter(lambda p: p.quality > 0.25, self.points)
        clusters = find_clusters(acceptable, max_length = 3)

        self.frame_count += 1

        self.frames.put((self.name, self.frame_count, frame, points, clusters))

    def get_frame(self):
        """
        Gets the frame in the queue. This is equivalent to `self.frames.get()`.
        :return: name, frame_count, frame, points, clusters
        """

        return self.frames.get()

    def process_dummy(self, frame, iterations = 1):
        last = None

        for i in range(iterations):
            self.process(frame)
            last = self.get_frame()

        return last

    def run(self):
        self.running = True

        while self.running:
            ret, frame = self.capture.read()

            self.process(frame)

        self.capture.release()


def show_camera((name, frame_count, frame, points, clusters)):
    dest = frame.copy()

    for point in points:
        point.draw(dest, frame_count)

    #for cluster, center in clusters:
    #    for point in cluster:
    #        cv2.line(dest, point.pos, center.pos, (0, 255, 0), 1)

    #    cv2.circle(dest, center.pos, 4, (255, 255, 255), 2)

    cv2.putText(dest, "Frame #%d" % frame_count,
                (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), lineType = cv2.LINE_AA)

    cv2.imshow(name, dest)


def show_dummy(thread, frame, iterations = 1):
    """
    Processes the given frame in the given TrackingThread for some number of
    iterations. Useful for simulating against a still image when a tracking
    history is needed.

    :param thread: a TrackingThread instance
    :param frame: the frame to process
    :param iterations: the number of iterations to process the same frame
    """
    while True:
        show_camera(thread.process_dummy(frame, iterations))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(thread.name)

    for i in range(5):
        cv2.waitKey(1) # wat


def main():
    threads = [
        TrackingThread(0, "Center"),
        #TrackingThread(1, "Right"),
        #TrackingThread(2, "Left")
    ]

    for thread in threads:
        thread.start()

    while True:
        for thread in threads:
            show_camera(thread.frames.get())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for thread in threads:
        thread.running = False
        thread.frames.get()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

