import numpy as np
import cv2

def Harris(img, k, wd, threshold):

    # find x and y derivatives for each pixel
    # usel Sobel filter

    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=wd)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=wd)

    # find dx^2, dy^2, dxy

    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy

    # apply the window
    # get the summations

    window_size = 5
    
    window = np.ones((window_size, window_size))
    dx2_sum = cv2.filter2D(dx2, -1, window)
    dy2_sum = cv2.filter2D(dy2, -1, window)
    dxy_sum = cv2.filter2D(dxy, -1, window)

    # Harris response
    # R = lambda1*lambda2 - k(lambda1+lambda2)^2
    # R = det(C) - k(Tr(C))^2
    # C = [[dx2_sum, dxy_sum],[dxy_sum,dy2_sum]] (2x2)
    # no need to construct C, just find Det and Tr

    det = dx2_sum * dy2_sum - dxy_sum * dxy_sum
    tr = dx2_sum + dy2_sum
    harris_response = det - k * (tr ** 2)

    # Normalize harris response (0-255)
    
    harris_response_norm = cv2.normalize(harris_response, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # apply threshold

    threshold = threshold * harris_response.max()
    keypoints = np.argwhere(harris_response > threshold)


    return harris_response_norm, keypoints