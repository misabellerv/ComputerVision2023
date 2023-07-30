import numpy as np
import cv2

# read the images

q1 = cv2.imread('C:/Users/Isabelle/Desktop/computervision/Listas_e_dados/Listas_e_dados/CV_Lista01/CV_Lista01_dados/Quadros_01.jpg')
q2 = cv2.imread('C:/Users/Isabelle/Desktop/computervision/Listas_e_dados/Listas_e_dados/CV_Lista01/CV_Lista01_dados/Quadros_02.jpg')
q3 = cv2.imread('C:/Users/Isabelle/Desktop/computervision/Listas_e_dados/Listas_e_dados/CV_Lista01/CV_Lista01_dados/Quadros_03.jpg')
ref = cv2.imread('C:/Users/Isabelle/Desktop/computervision/Listas_e_dados/Listas_e_dados/CV_Lista01/CV_Lista01_dados/Quadros_ref.jpg')

# reference points

ref_pts = [(0,0), (0,91.5),(238.4, 91.5), (238.4,0)]

# first image

def p1(event, x, y, flags, param):
    global counter, q1, coords1

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)
        cv2.circle(q1, point, 3, (0, 0, 255), -1)
        if counter < 4:
            coords1.append(point)
        counter += 1
        if counter == 4:
            cv2.imshow("points", q1)

gray = cv2.cvtColor(q1,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("points")
counter = 0
coords1 = [] 

cv2.setMouseCallback("points", p1)
cv2.imshow("points", q1)
while cv2.waitKey(0) != 27:
    pass
cv2.destroyAllWindows()
cv2.imwrite('q1_points.png',q1)

coords1 = np.array(coords1)
ref_pts = np.array(ref_pts)

# find homography using openCV automatic function

def findHomography(p1, p2):
    A = []
    N = len(p1)
    for i in range(N):
        x1, y1 = p1[i][0], p1[i][1]
        x2, y2 = p2[i][0], p2[i][1] 
        A.append([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2])
        A.append([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2])
    _, _, V = np.linalg.svd(np.asarray(A))
    H = V[-1, :] / V[-1, -1]
    H = H.reshape(3, 3)
    return H

H1 = findHomography(coords1, ref_pts)

# apply homography 

height, width, _ = q1.shape
new_image_size = (width, height)

correction1 = cv2.warpPerspective(q1, H1, new_image_size)

side_by_side1 = np.hstack((q1, correction1))

cv2.imshow('Original image and corrected image', side_by_side1)
cv2.imwrite('q1_corrected.png', side_by_side1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# second image

def p2(event, x, y, flags, param):
    global counter, q2, coords2

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)
        cv2.circle(q2, point, 3, (0, 0, 255), -1)
        if counter < 4:
            coords2.append(point)
        counter += 1
        if counter == 4:
            cv2.imshow("points", q2)

gray = cv2.cvtColor(q2,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("points")
counter = 0
coords2 = [] 

cv2.setMouseCallback("points", p2)
cv2.imshow("points", q2)

while cv2.waitKey(0) != 27:
    pass
cv2.destroyAllWindows()
cv2.imwrite('q2_points.png',q2)

coords2 = np.array(coords2)

# find homography using openCV automatic function

H2 = findHomography(coords2, ref_pts)

# apply homography 

height, width, _ = q2.shape
new_image_size = (width, height)

correction2 = cv2.warpPerspective(q2, H2, new_image_size)

side_by_side2 = np.hstack((q2, correction2))

cv2.imshow('Original image and corrected image', side_by_side2)
cv2.imwrite('q2_corrected.png', side_by_side2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# third image

def p3(event, x, y, flags, param):
    global counter, q3, coords3

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)
        cv2.circle(q3, point, 3, (0, 0, 255), -1)
        if counter < 4:
            coords3.append(point)
        counter += 1
        if counter == 4:
            cv2.imshow("points", q3)

gray = cv2.cvtColor(q3,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("points")
counter = 0
coords3 = [] 

cv2.setMouseCallback("points", p3)
cv2.imshow("points", q3)
while cv2.waitKey(0) != 27:
    pass
cv2.destroyAllWindows()
cv2.imwrite('q3_points.png',q3)

coords3 = np.array(coords3)

# find homography using openCV automatic function

H3 = findHomography(coords2, ref_pts)

# apply homography 

height, width, _ = q3.shape
new_image_size = (width, height)

correction3 = cv2.warpPerspective(q3, H3, new_image_size)

side_by_side3 = np.hstack((q3, correction3))

cv2.imshow('Original image and corrected image', side_by_side3)
cv2.imwrite('q3_corrected.png', side_by_side3)
cv2.waitKey(0)
cv2.destroyAllWindows()