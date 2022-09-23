# Overview
## Algorithm
`alproj` is a simple python package for geo-rectification of alpine landscape photographs. 
`alproj` has 4 steps.
1. Setting Ground Control Points (GCPs) in target photographs, using simulated landscape images rendered with Digital Surface Models and airborne photographs.
2. Heuristic estimation of camera parameters including the camera angle, field of view, and lens distortions (shooting point of the photograph is required).
3. Perspective reverse projection of the target photograph on Digital Surface Model, with estimated camera parameters, using OpenGL.

## Future applications in alpine ecology, geology and glaciology
