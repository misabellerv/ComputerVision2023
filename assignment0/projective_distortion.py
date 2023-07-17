import cv2
import numpy as np
from matplotlib import pyplot as plt

#### removing projective distortion ####
# this is part 1 from affine rectification

# functions 

def callback(event, x, y, flags, param):
    global counter, img, coords1, coords2

    if event == cv2.EVENT_LBUTTONDOWN: # if left clicking occurs
        point = (x, y)
        cv2.circle(img, point, 3, (0, 0, 255), -1)
        if click_counter < 4: # first coordinate list
            coords1.append(point)
        elif 4 <= click_counter <= 8:
            coords2.append(point)
        click_counter += 1
        if click_counter == 8: # draw the lines
            cv2.line(img, coords1[0], coords1[1], (0, 0, 255), 2) 
            cv2.line(img, coords1[2], coords1[3], (0, 0, 255), 2)
            cv2.line(img, coords2[0], coords2[1], (0, 255, 0), 2)
            cv2.line(img, coords2[2], coords2[3], (0, 255, 0), 2)
            cv2.imshow("straight lines", img)
            #cv2.imwrite('C:/Users/Isabelle/Desktop/computervision/lista0/img_lines.png',img)

# read the image

img = cv2.imread('C:/Users/Isabelle/Desktop/computervision/lista0/img_retificacao.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("straight lines")

# define a list for each pair or lines

counter = 0
coords1 = [] # 1st pair of parallel lines
coords2 = [] # 2nd pair of parallel lines

# select coordinates from parallel lines

cv2.setMouseCallback("straight lines", callback)
cv2.imshow("straight lines", img)
while cv2.waitKey(0) != 27:
    pass
cv2.destroyAllWindows()
print(f'first pair coordinates: {coords1}')
print(f'second pair coordinates: {coords2}')

# find line equations
# use cross product l = x1 x x2
# don't forget homogeneous coordinates

# homogenous coordinates (first pair)

point1 = np.array([coords1[0][0], coords1[0][1], 1])
point2 = np.array([coords1[1][0], coords1[1][1], 1])
point3 = np.array([coords1[2][0], coords1[2][1], 1])
point4 = np.array([coords1[3][0], coords1[3][1], 1])

# homogenous coordinates (second pair)

point5 = np.array([coords2[0][0], coords2[0][1], 1])
point6 = np.array([coords2[1][0], coords2[1][1], 1])
point7 = np.array([coords2[2][0], coords2[2][1], 1])
point8 = np.array([coords2[3][0], coords2[3][1], 1])

# cross product

l1 = np.cross(point1, point2)
l2 = np.cross(point3, point4)
l3 = np.cross(point5, point6)
l4 = np.cross(point7, point8)

# vanishing points

p1 = np.cross(l1, l2)
p2 = np.cross(l3, l4)

# homogeneous vanishing points

p1 = p1/p1[2]
p2 = p2/p2[2]

# infinity line points (l1,l2,l3)

l = np.cross(p1, p2)

# matrix H1 with homogeneous coords from l

H1 = np.float32(np.array([[1, 0, 0], [0, 1, 0], [l[0]/l[2], l[1]/l[2],1]]))

# apply transform 

size = gray.shape
sizeNew = (size[1], size[0])
#H_inv = np.linalg.inv(H1)
AffineRect = cv2.warpPerspective(img,H1,sizeNew)
cv2.imshow("Affine Rectification", AffineRect)
#cv2.imwrite("C:/Users/Isabelle/Desktop/computervision/lista0/AffineRectifiedImage.jpg", AffineRect)
while cv2.waitKey(0) != 27:
    pass