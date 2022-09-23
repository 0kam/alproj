# alproj
A simple georectification tool for alpine landscape photographs.

[Documentation](https://alproj.readthedocs.io/en/latest/index.html)
[日本語ドキュメント](https://green-light.netlify.app/posts/usage_of_alproj/)
## Overview
`alproj` is a simple python package for geo-rectification of alpine landscape photographs.   
`alproj` has 3 steps for geo-rectification of landscape photograph.  

1. Setting Ground Control Points (GCPs) in target photographs, using simulated landscape images rendered with Digital Surface Models and airborne photographs.
![](docs/_static/setting_up_gcps.jpg)

2. Heuristic estimation of camera parameters including the camera angle, field of view, and lens distortions (shooting point of the photograph is required).
![](docs/_static/estimation_of_camera_parameters.jpg)

3. Perspective reverse projection of the target photograph on Digital Surface Model, with estimated camera parameters, using OpenGL.
![](docs/_static/georectification.jpg)

![](docs/_static/ortholike.png)

This project aims to revive alpine landscape photographs in your photo albams, as valuable geospatial data that may reveal the unknown changes of alpine landscape, ecosystem and cryosphere!

## Installation
1. Make sure that you are using the latest version of pip
2. Run `pip install git+https://github.com/0kam/alproj`

`alproj` is depending on `datatable` library and `datatable` installation fails in python 3.10 environment. If you encountered this, please install `datatable` from github repository. 
Run `pip install git+https://github.com/h2oai/datatable`.

## Acknowledgements
The example photograph is of [NIES' long-period monitoring](https://db.cger.nies.go.jp/gem/ja/mountain/station.html?id=2).   