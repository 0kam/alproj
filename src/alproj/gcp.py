import cv2
import numpy as np
import pandas as pd

def akaze_match(path_org, path_sim, ransac_th=100, plot_result=False):
    """
    AKAZE matching between the original (real) photograph and a simulated landscape image 
    The work flow is shown below.
    - AKAZE local feature detection
    - FLANN matching
    - Find and remove outliers by homography transformation with RANSAC

    Parameters
    ----------
    path_org : str
        Path for original photograph
    path_sim : str
        Path for simulated landscape image
    ransac_th : int default 100
        If the error (pixel) of homography transformation on a point pair is larger than this value, it will be removed as an outlier.
    plot_result : boolean default False
        Whether return a result plot
    
    Returns 
    -------
    points : pd.DataFrame
        The coordinates of matched points. (Left-Top origin)
    plot : np.array
        An OpenCV image of result plot.
    """
    im_org = cv2.imread(path_org)
    im_sim = cv2.imread(path_sim)
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(im_org, None)
    kp2, des2 = akaze.detectAndCompute(im_sim, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    ratio = 0.8
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    pts1 = np.float32([ kp1[match.queryIdx].pt for match in good ])
    pts2 = np.float32([ kp2[match.trainIdx].pt for match in good ])
    pts1 = pts1.reshape(-1,1,2)
    pts2 = pts2.reshape(-1,1,2)
    
    # Filter matched points with RANSAC 
    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_th)
    pts1 = pts1[mask.astype('bool')].astype("int32")
    pts2 = pts2[mask.astype('bool')].astype("int32")
    pts = pd.DataFrame(np.hstack((pts1, pts2)), columns=["u_org","v_org","u_sim","v_sim"])
    if plot_result:
        for i in range(pts1.shape[0]):
            im_org = cv2.arrowedLine(im_org, tuple(pts1[i,:]), tuple(pts2[i,:]), color = [180,105,255], thickness=20, tipLength=0.3)
        im_org = cv2.putText(im_org, "simulated image <- original image", (int(im_org.shape[1]*0.15), int(im_org.shape[0]*0.05)),\
             cv2.FONT_HERSHEY_TRIPLEX, 5, (0,0,0), 5, cv2.LINE_AA)
        return {"points":pts, "plot":im_org}
    else:
        return {"points":pts, "plot":None}


def set_gcp(match, rev_proj):
    """
    Adds giographic coordinates to the matched point pairs.
    The result of this function will be used as the Ground Control Points (GCPs) 
    during camera parameter estimation

    Parameters
    ----------
    match : pd.DataFrame
        Result of alproj.gcp.akaze_match()
    rev_proj : pd.DataFrame
        Result of alproj.project.rverse_proj
    
    Returns
    -------
    gcp : pd.DataFrame
        A dataframe with 4 columns
        - u : x_axis coordinates of the Ground Control Points on the original photograph. Left-Top origin.
        - v : y_axis coordinates of the GCPs.
        - x : X coordinates of GCPs in a (planer) giographic coordinate system.
        - y : Y coordinates of GCPs.
        - z : Z coordinates of GCPs. 
    """
    gcp = pd.merge(match, rev_proj, how="left",\
         left_on=["u_sim", "v_sim"], right_on=["u", "v"]) \
             [["u_org","v_org","x","y","z"]] \
                 .rename(columns={"u_org":"u", "v_org":"v"})
    return gcp
