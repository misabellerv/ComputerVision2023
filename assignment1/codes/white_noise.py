import numpy as np
import matplotlib.pyplot as plt

# 2d points

p12D = np.load('C:/Users/Isabelle/Desktop/computervision/Listas_e_dados/Listas_e_dados/CV_Lista01/CV_Lista01_dados/Pontos2D_1.npy')
p22D = np.load('C:/Users/Isabelle/Desktop/computervision/Listas_e_dados/Listas_e_dados/CV_Lista01/CV_Lista01_dados/Pontos2D_2.npy')

# 3d points

p13D = np.load('C:/Users/Isabelle/Desktop/computervision/Listas_e_dados/Listas_e_dados/CV_Lista01/CV_Lista01_dados/Pontos3D_1.npy')
p23D = np.load('C:/Users/Isabelle/Desktop/computervision/Listas_e_dados/Listas_e_dados/CV_Lista01/CV_Lista01_dados/Pontos3D_2.npy')

# add white noise to 3D points 
# σ = 0.05 ∗ max3D
# max3D = maximum absolute value of 3D coord points

max_3D1 = np.max(np.abs(p13D))
max_3D2 = np.max(np.abs(p23D))

sigma1 = 0.05 * max_3D1
sigma2 = 0.05 * max_3D2

random_noise1 = np.random.normal(loc=0, scale=sigma1, size=p13D.shape)
random_noise2 = np.random.normal(loc=0, scale=sigma2, size=p23D.shape)


p13Dnoisy = p13D + random_noise1
p23Dnoisy = p23D + random_noise2

# create 3D figures for each pair 2D-3D

# first pair

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x1 = p13Dnoisy[:, 0]
y1 = p13Dnoisy[:, 1]
z1 = p13Dnoisy[:, 2]
ax.scatter(x1, y1, z1, c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D_1.npy + white noise')
plt.savefig('3d1_noisy.png')
plt.show()

# second pair

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x2 = p23Dnoisy[:, 0]
y2 = p23Dnoisy[:, 1]
z2 = p23Dnoisy[:, 2]
ax.scatter(x2, y2, z2, c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D_2.npy + white noise')
plt.savefig('3d2_noisy.png')
plt.show()

# find P matrix for the first pair

# A1 is 2*N1 x 12 since we have N1 points but 2 x and y equations for each two lines
# white noise is not added to 2D points

N1 = p12D.shape[0]
A1 = np.zeros((2*N1, 12))

i = 0

while True:
    xi, yi, _ = p12D[i]
    Xi, Yi, Zi, _ = p13Dnoisy[i]
    A1[i] = [Xi, Yi, Zi, 1, 0, 0, 0, 0, -xi*Xi, -xi*Yi, -xi*Zi, -xi]
    A1[1+i] = [0, 0, 0, 0, Xi, Yi, Zi, 1, -yi*Xi, -yi*Yi, -yi*Zi, -yi]
    i +=2
    if i == N1+1:
        break

# SVD decomposition to solve Ap = 0

U1, S1, V1 = np.linalg.svd(A1)
p1 = V1[-1]

# p is 12x1
# Reshape p -> P 3x4
P1 = p1.reshape(3, 4)

print(f'Projection matrix from first pair of 2D-3D coordinates (noisy):\n{P1}')

# find P matrix for the second pair

# A2 is 2*N2 x 12 since we have N2 points but 2 x and y equations for each two lines

N2 = p22D.shape[0]
A2 = np.zeros((2*N2, 12))

i = 0

while True:
    xi, yi, _ = p22D[i]
    Xi, Yi, Zi, _ = p23Dnoisy[i]
    A2[i] = [Xi, Yi, Zi, 1, 0, 0, 0, 0, -xi*Xi, -xi*Yi, -xi*Zi, -xi]
    A2[1+i] = [0, 0, 0, 0, Xi, Yi, Zi, 1, -yi*Xi, -yi*Yi, -yi*Zi, -yi]
    i +=2
    if i == N2+1:
        break

# SVD decomposition to solve Ap = 0

U2, S2, V2 = np.linalg.svd(A2)
p2 = V2[-1]

# p is 12x1
# Reshape p -> P 3x4
P2 = p2.reshape(3, 4)

print(f'Projection matrix from second pair of 2D-3D coordinates (noisy):\n{P2}')

# check if your matrices are correct using projection error

# for the first pair of points

projected_points1 = np.dot(p13Dnoisy, P1.T)

# Normalize projected points

projected_points_normalized1 = projected_points1[:, :2] / projected_points1[:, 2:]

errors1 = np.linalg.norm(p12D[:, :2] - projected_points_normalized1, axis=1)

print("Erros de projeção para cada ponto (primeiro par 2D-3D) (noisy):")
print(errors1)

plt.figure()
plt.scatter(p12D[:, 0] / p12D[:, 2], p12D[:, 1] / p12D[:, 2], c='r', marker='o', label='Real points')
plt.scatter(projected_points_normalized1[:, 0], projected_points_normalized1[:, 1], c='b', marker='x', label='Projected points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Real x Projected points (1st 2D-3D) (noisy)')
plt.grid()
plt.savefig('projection_error1_noisy.png')
plt.show()

# for the second pair of points

projected_points2 = np.dot(p23Dnoisy, P2.T)

# Normalize projected points

projected_points_normalized2 = projected_points2[:, :2] / projected_points2[:, 2:]

errors2 = np.linalg.norm(p22D[:, :2] - projected_points_normalized2, axis=1)

print("Erros de projeção para cada ponto (Segundo par 2D-3D) (noisy):")
print(errors2)

plt.figure()
plt.scatter(p22D[:, 0] / p22D[:, 2], p22D[:, 1] / p22D[:, 2], c='r', marker='o', label='Real points')
plt.scatter(projected_points_normalized2[:, 0], projected_points_normalized2[:, 1], c='b', marker='x', label='Projected points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Real x Projected points (2nd 2D-3D) (noisy)')
plt.grid()
plt.savefig('projection_error2_noisy.png')
plt.show()