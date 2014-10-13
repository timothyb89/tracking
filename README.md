tracking
========

Cheap and easy point and hand tracking in Python with OpenCV.

Goals
-----
* Inexpensive: most hand tracking products currently available are expensive, difficult to obtain, etc. This uses less than $50 in off-the-shelf parts which you can acquire at a hardware store.
* Unpowered: no IR LEDs or other "active" tracking aids needed needed
* Simple: no complex calibration required, can utilize an arbitrary number of cameras

What's the catch?
-----------------
A few requirements on the environment are made to make this feasible.

1. Points are tracked on a black surface
2. 3 points must be arranged on each trackable surface to form an equlateral triangle
3. Point clusters should be colored uniquely
 
Ideally, you can use a pair of black gloves and some stickers as a quick tracking surface. You can potentially use points on either side of the hand to reduce the necessary camera coverage, though the back of the hand is a lot easier to rely on.

Recommended Hardware
--------------------
* 2x [PlayStation Eye cameras](http://www.amazon.com/gp/product/B0072I2240?ie=UTF8&tag=blogccasion-20&link_code=as2&camp=212361&creative=380601&creativeASIN=B0072I2240), ~$8x2 = $16 shipped
  * Only 1 is needed, but multiple are useful to ensure visual coverage and prevent obstruction
* [Circle stickers](http://www.amazon.com/gp/product/B002EJ6P40?ie=UTF8&tag=blogccasion-20&link_code=as2&camp=212361&creative=380601&creativeASIN=B002EJ6P40), ~$7
  * [White stickers](http://www.amazon.com/gp/product/B00KSTAGL6?ie=UTF8&tag=blogccasion-20&link_code=as2&camp=212361&creative=380601&creativeASIN=B00KSTAGL6) are useful as well
* Any pair of black gloves

What works?
-----------
Currently, vision processing, point tracking, and cluster determination appear to be working (and performing) well. Mapping of points into 3d space is still a WIP.

TODO
----
* Mapping of 2d points into 6 DoF 3d space (WIP)
* A user interface with calibration tools, editable tracking parameters, etc.
* More documentation
