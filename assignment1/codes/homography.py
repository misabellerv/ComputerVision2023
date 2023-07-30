import numpy as np
import cv2

# txt files

h1 = np.loadtxt('C:/Users/Isabelle/Desktop/computervision/Listas_e_dados/Listas_e_dados/CV_Lista01/CV_Lista01_dados/homografia_1.txt')
h2 = np.loadtxt('C:/Users/Isabelle/Desktop/computervision/Listas_e_dados/Listas_e_dados/CV_Lista01/CV_Lista01_dados/homografia_2.txt')

# img files

img1 = cv2.imread('C:/Users/Isabelle/Desktop/computervision/Listas_e_dados/Listas_e_dados/CV_Lista01/CV_Lista01_dados/img_homografia_1.png')
img2 = cv2.imread('C:/Users/Isabelle/Desktop/computervision/Listas_e_dados/Listas_e_dados/CV_Lista01/CV_Lista01_dados/img_homografia_2.png')

# first array column denotes Y and second X. flip these to get (x,y)

h1[:, [0, 1]] = h1[:, [1, 0]]
h2[:, [0, 1]] = h2[:, [1, 0]]

h1 = h1.astype(int)
h2 = h2.astype(int)

# check the images

for point in h1:
    cv2.circle(img1, point, 1, (0, 0, 255), -2)  
cv2.imshow('Image 1 with points', img1)
cv2.imwrite('img1_points.png', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

for point in h2:
    cv2.circle(img2, point, 1, (0, 0, 255), -2) 
cv2.imshow('Image 2 with points', img2)
cv2.imwrite('img2_points.png', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

H21 = findHomography(h2,h1)

# Definir o tamanho da nova imagem após a transformação
height, width, _ = img1.shape
new_image_size = (width, height)

# Aplicar a matriz de homografia H21 na imagem usando cv2.warpPerspective()
warped_image = cv2.warpPerspective(img2, H21, new_image_size)

# Mostrar a imagem transformada
cv2.imshow('H21', warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Concatenar as três imagens horizontalmente
side_by_side = np.hstack((img1, img2, warped_image))

# Mostrar as imagens lado a lado
cv2.imshow('Image 1, Image 2 and transformed image (H21)', side_by_side)
cv2.imwrite('h21.png', side_by_side)
cv2.waitKey(0)
cv2.destroyAllWindows()


H12 = findHomography(h1,h2)

# Definir o tamanho da nova imagem após a transformação
height, width, _ = img2.shape
new_image_size = (width, height)

# Aplicar a matriz de homografia H12 na imagem usando cv2.warpPerspective()
warped_image2 = cv2.warpPerspective(img1, H12, new_image_size)

# Mostrar a imagem transformada
cv2.imshow('H12', warped_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Concatenar as três imagens horizontalmente
side_by_side = np.hstack((img1, img2, warped_image2))

# Mostrar as imagens lado a lado
cv2.imshow('Image 1, Image 2 and transformed image (H12)', side_by_side)
cv2.imwrite('h12.png', side_by_side)
cv2.waitKey(0)
cv2.destroyAllWindows()
