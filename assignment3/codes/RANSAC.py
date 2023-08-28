import cv2
import matplotlib.pyplot as plt
import numpy as np

def RANSAC(N, threshold, matches, kp1, kp2):
    "Returns best Homography H"

    best_H = None
    best_inliers = []
    errors = []

    for i in range(N):

        # choosing 4 random correspondences
        random_matches = np.random.choice(matches, 4, replace=False)

        # src_pts: array coords first img (randomly chosen)
        # dst_pts: array destination img 
        # We use them to estimate homography between images
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in random_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in random_matches]).reshape(-1, 1, 2)
        
        # Homography 
        H, _ = cv2.findHomography(src_pts, dst_pts)
        
        inliers = []
        
        # Projection error and inlier checking
        for match in matches:
            src_pt = kp1[match.queryIdx].pt
            dst_pt = kp2[match.trainIdx].pt
            
            projected_pt = np.dot(H, np.array([src_pt[0], src_pt[1], 1]))
            projected_pt /= projected_pt[2]
            
            error = np.sqrt((projected_pt[0] - dst_pt[0])**2 + (projected_pt[1] - dst_pt[1])**2)
            errors.append(error)
            
            if error < threshold:
                inliers.append(match)
        
        if len(inliers) > len(best_inliers):
            best_H = H
            best_inliers = inliers

        src_in = np.float32([kp1[m.queryIdx].pt for m in best_inliers]).reshape(-1, 1, 2)
        dst_in = np.float32([kp2[m.trainIdx].pt for m in best_inliers]).reshape(-1, 1, 2)

        final_H, _ = cv2.findHomography(src_in,dst_in)
        
    return final_H, errors
