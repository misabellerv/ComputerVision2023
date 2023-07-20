import numpy as np
import cv2

### one-step method image rectification

# get pure image

affine_img = cv2.imread('C:/Users/Isabelle/Desktop/computervision/lista0/parallel_afine/img_retificacao.png')
gray = cv2.cvtColor(affine_img,cv2.COLOR_BGR2GRAY)

# select 5 pairs of ortogonal lines

def callback(event, x, y, flags, param):
    global counter, affine_img, perp1, perp2, perp3, perp4, perp5

    if event == cv2.EVENT_LBUTTONDOWN: # if left clicking occurs
        point = (x, y)
        cv2.circle(affine_img, point, 3, (0, 0, 255), -1)
        if counter < 4: # first coordinate list
            perp1.append(point)
        elif 4 <= counter < 8:
            perp2.append(point)
        elif 8 <= counter < 12:
            perp3.append(point) 
        elif 12 <= counter < 16:
            perp4.append(point)
        elif 16 <= counter < 20:
            perp5.append(point)
        counter += 1

cv2.namedWindow("perp lines")
perp1 = []
perp2 = []
perp3 = []
perp4 = []
perp5 = []
counter = 0

# get the points from perpendicular lines

cv2.setMouseCallback("perp lines", callback)
cv2.imshow("perp lines", affine_img)
while cv2.waitKey(0) != 27:
    pass
cv2.destroyAllWindows()

# get the points in homogeneous coordinates

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

# homogenous coordinates (third perpendicular pair)

ppoint9 = np.array([perp3[0][0], perp3[0][1], 1])
ppoint10 = np.array([perp3[1][0], perp3[1][1], 1])
ppoint11= np.array([perp3[2][0], perp3[2][1], 1])
ppoint12 = np.array([perp3[3][0], perp3[3][1], 1])

# homogenous coordinates (fourth perpendicular pair)

ppoint13 = np.array([perp4[0][0], perp4[0][1], 1])
ppoint14 = np.array([perp4[1][0], perp4[1][1], 1])
ppoint15= np.array([perp4[2][0], perp4[2][1], 1])
ppoint16 = np.array([perp4[3][0], perp4[3][1], 1])

# homogenous coordinates (fifth perpendicular pair)

ppoint17 = np.array([perp5[0][0], perp5[0][1], 1])
ppoint18 = np.array([perp5[1][0], perp5[1][1], 1])
ppoint19= np.array([perp5[2][0], perp5[2][1], 1])
ppoint20 = np.array([perp5[3][0], perp5[3][1], 1])

# cross product perpendicular
# find lines equations coefs

l1 = np.cross(ppoint1, ppoint2)
m1 = np.cross(ppoint3, ppoint4)

l2 = np.cross(ppoint5, ppoint6)
m2 = np.cross(ppoint7, ppoint8)

l3 = np.cross(ppoint9, ppoint10)
m3 = np.cross(ppoint11, ppoint12)

l4 = np.cross(ppoint13, ppoint14)
m4 = np.cross(ppoint15, ppoint16)

l5 = np.cross(ppoint17, ppoint18)
m5 = np.cross(ppoint19, ppoint20)

# homogeneous coordinates

l1 = l1/l1[2]
m1 = m1/m1[2]
l2 = l2/l2[2]
m2 = m2/m2[2]
l3 = l3/l3[2]
m3 = m3/m3[2]
l4 = l4/l4[2]
m4 = m4/m4[2]
l5 = l5/l5[2]
m5 = m5/m5[2]

# solve the system for (a,b,c,d,e) from the conic

# [l[0]*m[0], 0.5*(l[0]*m[1]+l[1]*m[0]), l[1]*m[1], 0.5*(l[0]*m[2]+l[2]*m[0]), 0.5*(l[1]*m[2]+l[2]*m[1]), l[2]*m[2]]*v = 0

m1 = [l1[0]*m1[0], 0.5*(l1[0]*m1[1]+l1[1]*m1[0]), l1[1]*m1[1], 0.5*(l1[0]*m1[2]+l1[2]*m1[0]), 0.5*(l1[1]*m1[2]+l1[2]*m1[1])] 
s1 = [-l1[2]*m1[2]]
m2 = [l2[0]*m2[0], 0.5*(l2[0]*m2[1]+l2[1]*m2[0]), l2[1]*m2[1], 0.5*(l2[0]*m2[2]+l2[2]*m2[0]), 0.5*(l2[1]*m2[2]+l2[2]*m2[1])] 
s2 = [-l2[2]*m2[2]]
m3 = [l3[0]*m3[0], 0.5*(l3[0]*m3[1]+l3[1]*m3[0]), l3[1]*m3[1], 0.5*(l3[0]*m3[2]+l3[2]*m3[0]), 0.5*(l3[1]*m3[2]+l3[2]*m3[1])] 
s3 = [-l3[2]*m3[2]]
m4 = [l4[0]*m4[0], 0.5*(l4[0]*m4[1]+l4[1]*m4[0]), l4[1]*m4[1], 0.5*(l4[0]*m4[2]+l4[2]*m4[0]), 0.5*(l4[1]*m4[2]+l4[2]*m4[1])] 
s4 = [-l4[2]*m4[2]]
m5 = [l5[0]*m5[0], 0.5*(l5[0]*m5[1]+l5[1]*m5[0]), l5[1]*m5[1], 0.5*(l5[0]*m5[2]+l5[2]*m5[0]), 0.5*(l5[1]*m5[2]+l5[2]*m5[1])] 
s5 = [-l5[2]*m5[2]]

S = np.array([s1,s2,s3,s4,s5])
M = np.array([m1,m2,m3,m4,m5])

s = np.matmul(np.linalg.inv(M),S)

# find the values for a, b, c, d, e
# we set f=1

print(f's: {s}')

a = s[0][0]
b = s[1][0]
c = s[2][0]
d = s[3][0]
e = s[4][0]

# plug a,b,c,d,e in matrix C

C = np.array([[a, b/2, d/2],[b/2, c, e/2],[d/2, e/2, 1]])

# do SVD decomposition

U, D, V = np.linalg.svd(C)

# find C eigenvalues

eigenvalues = np.linalg.eigvalsh(C)

print("eigenvalues:")
print(eigenvalues)

'''
# H = UD

H = U*D

# Restored image = H^-1 Distorted image

Hinv = np.float32(np.linalg.inv(H)) 

size = gray.shape
sizeNew = (size[1], size[0])
MetricRect = cv2.warpPerspective(affine_img,Hinv, sizeNew)

cv2.imshow("Metric Rectification", MetricRect)
#cv2.imwrite("MetricConic.jpg", MetricRect)
while cv2.waitKey(0) != 27:
    pass
cv2.destroyAllWindows()
cv2.waitKey(1)'''

