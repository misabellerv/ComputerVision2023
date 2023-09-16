import numpy as np
from scipy.linalg import rq

import numpy as np
import cv2
import matplotlib.pyplot as plt

# cube 3D coordinates
p3D = np.array([
    [2, 2, 2],
    [-2, 2, 2],
    [-2, 2, -2],
    [2, 2, -2],
    [2, -2, 2],
    [-2, -2, 2],
    [-2, -2, -2],
    [2, -2, -2]
])

# image corresponding 2D coordinates
p2D = np.array([
    [422, 323],
    [178, 323],
    [118, 483],
    [482, 483],
    [438, 73],
    [162, 73],
    [78, 117],
    [522, 117]
])

# number of points 
N1 = p2D.shape[0]

## lets use direct linear method

# solve Ap = 0 instead of x = PX
# p =[p11 p12 ....]

A = np.zeros((2 * N1, 12))

for i in range(N1):
    xi, yi = p2D[i]
    Xi, Yi, Zi = p3D[i]
    A[i * 2] = [Xi, Yi, Zi, 1, 0, 0, 0, 0, -xi * Xi, -xi * Yi, -xi * Zi, -xi]
    A[i * 2 + 1] = [0, 0, 0, 0, Xi, Yi, Zi, 1, -yi * Xi, -yi * Yi, -yi * Zi, -yi]

# SVD decomposition
U1, S1, V1 = np.linalg.svd(A)
p1 = V1[-1]

P = p1.reshape(3, 4)
# left block of 'p'
p = P[:3, :3]


# RQ decomposition 'p'
K, R = rq(P[:,:3])

# normalize K 
K /= K[2, 2]

# make sure R determinant is +
if np.linalg.det(R) < 0:
    R *= -1

# make diagonal of K positive
T = np.diag(np.sign(np.diag(K)))
if np.linalg.det(T) < 0:
   T[0,1] *= -1
   K = np.dot(K,T)
   R = np.dot(T,R) # T is its own inverse
   t = np.dot(np.linalg.inv(K),P[:,3])
print(f'P = {P}')
print(f'K = {K}')
print(f'R = {R}')
print(f't = {t}')
