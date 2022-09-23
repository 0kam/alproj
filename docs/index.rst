.. alproj documentation master file, created by
   sphinx-quickstart on Fri Feb 12 09:19:44 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to alproj's documentation!
==================================
`alproj` is a simple python package for geo-rectification of alpine landscape photographs. 

`alproj` has 3 steps for geo-rectification of a landscape photograph.

1. Setting Ground Control Points (GCPs) in target photographs, using simulated landscape images rendered with Digital Surface Models and airborne photographs.

2. Heuristic estimation of camera parameters including the camera angle, field of view, and lens distortions (shooting point of the photograph is required).

3. Perspective reverse projection of the target photograph on Digital Surface Model, with estimated camera parameters, using OpenGL.

This project aims to revive alpine landscape photographs in your photo albams, as valuable geospatial data that may reveal the unknown changes of alpine landscape, ecosystem and cryosphere!


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   overview
   usage
   
.. toctree::
   :maxdepth: 2
   :caption: Package Reference:
   
   alproj.surface
   alproj.project
   alproj.gcp
   alproj.optimize
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`