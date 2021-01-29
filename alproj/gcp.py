import cv2
import numpy as np
import pandas as pd

def akaze_match(path_org, path_sim, ransac_th=100, plot_result=False):
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
        return pts, im_org
    else:
        return pts


def set_gcp(match, rev_proj):
    gcp = pd.merge(match, rev_proj, how="left",\
         left_on=["u_sim", "v_sim"], right_on=["u", "v"]) \
             [["u_org","v_org","x","y","z"]] \
                 .rename(columns={"u_org":"u", "v_org":"v"})
    return gcp
