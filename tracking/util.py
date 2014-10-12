# -*- coding: utf-8 -*-
"""
General utilities, mainly intended for use in an interactive (e.g. ipython) shell.
"""


def show(image, name = "Test"):
    """
    A short interactive debugging helper. Shows the given image in a frame, and
    waits for a 'q' keypress to quit.

    :param image: the image mat to show
    :param name: the name of the frame, 'Test' by default
    """
    while True:
        cv2.imshow(name, image)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(name)

    for i in range(5):
        cv2.waitKey(1) # wat
