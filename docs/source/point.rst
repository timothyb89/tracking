.. _Point:

Point Class and Tracking Implementation
***************************************

Point tracking is handled primarily within the :class:`tracking.point.Point` class.

Theory
======

Prediction Algorithm
--------------------

.. figure:: /images/point_search.png
    :align: center

Given some point with a history of at least 2 previous positions, we can determine the mean velocity over some window
(:data:`.WINDOW_SIZE`). Given this and the latest known position, we can then form a prediction line:

.. math::
    \newcommand\norm[1]{\left\lVert#1\right\rVert}

    \mathbf{a} =
    \begin{bmatrix}
        x \\
        y
    \end{bmatrix}, \
    \mathbf{n} =
    \begin{bmatrix}
        v_x \\
        v_y
    \end{bmatrix}, \
    \mathbf{n_{uv}} =
    \frac{
        \mathbf{n}
    }{
        \left\lVert
        \mathbf{n}
        \right\rVert
    }

    \mathbf{x} = \mathbf{a} + t\mathbf{n}

... where *t* is some number of number of frames into the future. Note that :math:`\mathbf{n_{uv}}` is the velocity
vector converted to a unit vector by dividing by it by its norm (used later).

To find a predicted position, we can simply plug in values to this function. For points being "actively" tracked, this
value will generally be 1; however, if tracking is temporarily lost for a point, this value should be the number of
frames since the point was last seen.

Similarly, we can manipulate the line to generate a probable bounding polygon for future point positions. A triangle
can be formed using some rotations and a translation:

.. math::
    \mathbf{n_{ccw}} =
    \begin{bmatrix}
        cos(\theta) & -sin(\theta) \\
        sin(\theta) &  cos(\theta)
    \end{bmatrix}
    \mathbf{n_{uv}}, \
    \mathbf{p_{ccw}} = \mathbf{a} + m
    \left\lVert \mathbf{n} \right\rVert
    \mathbf{n_{ccw}}

    \mathbf{n_{cw}} =
    \begin{bmatrix}
        cos(-\theta) & -sin(-\theta) \\
        sin(-\theta) &  cos(-\theta)
    \end{bmatrix}
    \mathbf{n_{uv}}, \
    \mathbf{p_{cw}} = \mathbf{a} + m
    \left\lVert \mathbf{n} \right\rVert
    \mathbf{n_{cw}}

    \mathbf{n_{rev}} = \mathbf{a} - m\mathbf{n_{uv}}

Constants:
 - :math:`\theta` - some arbitrary angle of rotation (see :data:`.ROTATION_THETA`)
 - :math:`m` - some arbitrary multiplier (see :data:`.BOUNDS_MULTIPLIER`)

Given the three points :math:`p_{rev}`, :math:`p_{ccw}`, and :math:`p_{cw}`, we can then use a point-in-polygon test
to determine if some point lies within the search bounds. Currently, this implementation uses
``cv2.pointPolygonTest()``.

From within these bounds, candidate points are then selected such that they globally minimize the distance from their
prediction. This ensures that pairing of candidate points to with known points is not dependent on discovery order and
that pairings will always be the best possible in any given dataset. (See: :func:`tracking.point.find_points`)

Other Factors
-------------
Stable tracking is highly dependent on hardware, lighting conditions, processing speed, and any number of other factors.
As such, a number of additional heuristics are applied to filter invalid points and to provide some level of immunity
to temporary tracking losses.

Point Health
^^^^^^^^^^^^
Point health is an integer value that is used to determine when points are considered "lost" and should be deleted.

The essentials are as follows:
 - New points are introduced with **0 health**
 - Points **gain 1** health point per frame they are successfully tracked
 - Points **lose 5** health points per frame when they are *not* successfully tracked
 - Points are **deleted** when they reach :data:`.FRAME_TIMEOUT`
 - Points have a maximum health of :data:`.POINT_MAX_HEALTH`

Note that points are created when a candidate point does not exist within the search bounds of an existing point, or
when all possible points within some search area are paired with better candidates.

Circularity
^^^^^^^^^^^
Circularity is a minor heuristic used primarily in point noise filtering and is a float :math:`0.0 \leq x \leq 1.0`.
See: :func:`tracking.vision.find_circles`.


Point Quality
^^^^^^^^^^^^^
Point quality is an additional metric primarily used for higher-level filtering, and is a ratio
:math:`0.0 \leq x \leq 1.0`. In particular it is based on:

 1. 75% ``max(0, health) / POINT_MAX_HEALTH``
 2. 25% ``circularity``

Uses include:
 - The UI may only display points over a certain value
 - The :ref:`cluster finding algorithm <Cluster>`

Implementation
==============

.. automodule:: tracking.point

Tunable Variables
-----------------
.. autodata:: tracking.point.BUFFER_LENGTH
.. autodata:: tracking.point.MAX_DISTANCE
.. autodata:: tracking.point.WINDOW_SIZE
.. autodata:: tracking.point.FRAME_TIMEOUT
.. autodata:: tracking.point.POINT_MAX_HEALTH
.. autodata:: tracking.point.VELOCITY_PREDICT_MINIMUM
.. autodata:: tracking.point.BOUNDS_MULTIPLIER
.. autodata:: tracking.point.ROTATION_THETA

``Point`` Class
---------------
.. autoclass:: tracking.point.Point
    :members:

``find_points()`` Function
--------------------------
.. autofunction:: tracking.point.find_points

Other Classes
-------------
.. autoclass:: tracking.point.SimplePoint
    :members:

