import numpy as np
import cv2

#### remove affine distortion ###
# this is part 2 from affine rectification

# start with the affine rectified image

# select 2 pairs of lines
# they must be perpendicular in the real world
# Metric Rectification
# new image is the affine rectified image

affine_img = cv2.imread('C:/Users/Isabelle/Desktop/computervision/lista0/parallel_afine/AffineRectifiedImage.jpg')
gray = cv2.cvtColor(affine_img,cv2.COLOR_BGR2GRAY)

def callback(event, x, y, flags, param):
    global counter, affine_img, perp1, perp2

    if event == cv2.EVENT_LBUTTONDOWN: # if left clicking occurs
        point = (x, y)
        cv2.circle(affine_img, point, 3, (0, 0, 255), -1)
        if counter < 4: # first coordinate list
            perp1.append(point)
        elif 4 <= counter <= 8:
            perp2.append(point)
        counter += 1
        if counter == 8: # draw the lines
            cv2.line(affine_img, perp1[0], perp1[1], (0, 0, 255), 2) 
            cv2.line(affine_img, perp1[2], perp1[3], (0, 0, 255), 2)
            cv2.line(affine_img, perp2[0], perp2[1], (0, 255, 0), 2)
            cv2.line(affine_img, perp2[2], perp2[3], (0, 255, 0), 2)
            cv2.imshow("perp lines", affine_img)
            #cv2.imwrite('C:/Users/Isabelle/Desktop/computervision/lista0/img_lines.png',img)
 
# create two list that contain the perpendicular lines
# same procedure as before...

cv2.namedWindow("perp lines")
perp1 = []
perp2 = []
counter = 0

# get the points from perpendicular lines

cv2.setMouseCallback("perp lines", callback)
cv2.imshow("perp lines", affine_img)
while cv2.waitKey(0) != 27:
    pass
cv2.destroyAllWindows()
print(f'first pair perpendicular coordinates: {perp1}')
print(f'second pair perpendicular coordinates: {perp2}')


# find l1, m1, and l2, m2 (pair of perpendicular lines)
# same procedure as before: cross product to find the lines coefs
# use homogeneous coordinates

# homogenous coordinates (first perpendicular pair)

ppoint1 = np.array([perp1[0][0], perp1[0][1], 1])
ppoint2 = np.array([perp1[1][0], perp1[1][1], 1])
ppoint3 = np.array([perp1[2][0], perp1[2][1], 1])
ppoint4 = np.array([perp1[3][0], perp1[3][1], 1])

# homogenous coordinates (second perpendicular pair)

ppoint5 = np.array([perp2[0][0], perp2[0][1], 1])
ppoint6 = np.array([perp2[1][0], perp2[1][1], 1])
ppoint7 = np.array([perp2[2][0], perp2[2][1], 1])
ppoint8 = np.array([perp2[3][0], perp2[3][1], 1])

# cross product perpendicular

l1 = np.cross(ppoint1, ppoint2)
m1 = np.cross(ppoint3, ppoint4)
l2 = np.cross(ppoint5, ppoint6)
m2 = np.cross(ppoint7, ppoint8)

# homogeneous coordinates

l1 = l1/l1[2]
m1 = m1/m1[2]
l2 = l2/l2[2]
m2 = m2/m2[2]

# now we have the perpendicular lines in homogeneous coordinates
# remember that l1 is perp to m1 and l2 is perp to m2

# system to solve: (l1m1, l1m2 + l2m1).T (s11 s12) = -l2m2
# Want to find the symmetric matrix S, s11 and s12 belongs to S
# (l1m1, l1m2 + l2m1).T (s11 s12) = -l2m2
# (l1'm1', l1'm2' + l2'm1').T (s11 s12) = -l2'm2'

M = np.array([[-m1[1]*l1[1]], [-m2[1]*l2[1]]])
B = np.array([[l1[0]*m1[0], l1[0]*m1[1] + l1[1]*m1[0]], [l2[0]*m2[0], l2[0]*m2[1] + l2[1]*m2[0]]])

s = np.matmul(np.linalg.inv(B),M)

#print(s)
#print(s[0])
#print(s[1])

# get symmetric matrix

S = np.array([[s[0][0], s[1][0]], [s[1][0], 1]])

print(f'Symmetric matrix: {S}')

# do general decomposition of symmetric matrix UDV

U, D, V = np.linalg.svd(S)

# since you have U,D,V, get the final results from affine distortions
# A = U sqrt(D) U.T
# U == V

Dsqrt = np.sqrt(D)

D = np.array([[Dsqrt[0], 0], [0, Dsqrt[1]]])
    
# A elements
a = np.matmul(np.matmul(U, D), V)

#Mult = np.matmul(np.sqrt(D), U.T)

H2 = np.array([[a[0][0], a[0][1], 0], [a[1][0], a[1][1], 0], [0, 0, 1]])

# apply transformation to image

inv_H2 = np.float32(np.linalg.inv(H2)) 

size = gray.shape
sizeNew = (size[1], size[0])
MetricRect = cv2.warpPerspective(affine_img,inv_H2, sizeNew)

cv2.imshow("Metric Rectification", MetricRect)
cv2.imwrite("MetricRectifiedImage.jpg", MetricRect)
while cv2.waitKey(0) != 27:
    pass
cv2.destroyAllWindows()
cv2.waitKey(1)