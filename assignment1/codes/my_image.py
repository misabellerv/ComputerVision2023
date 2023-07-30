import numpy as np
import cv2

img1 = cv2.imread('C:/Users/Isabelle/Desktop/computervision/lista1/images/nimona.jpg')

height, width = img1.shape[:2]
new_height = height // 2
new_width = width // 2

# Redimensionar a imagem para o novo tamanho
resized_image = cv2.resize(img1, (new_width, new_height))

img1 = resized_image

ref_points = [(0,0),(0,80),(264,80),(264,0)]

def p1(event, x, y, flags, param):
    global counter, img1, coords1

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)
        cv2.circle(img1, point, 3, (0, 0, 255), -1)
        if counter < 4:
            coords1.append(point)
        counter += 1
        if counter == 4:
            cv2.imshow("points", img1)

gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("points")
counter = 0
coords1 = [] 

cv2.setMouseCallback("points", p1)
cv2.imshow("points", img1)
#cv2.imwrite('q1_points.png',img1)
while cv2.waitKey(0) != 27:
    pass
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

H = findHomography(coords1,ref_points)

corrected_image = cv2.warpPerspective(img1, H, (new_width, new_height))

# Mostrar a imagem original e a imagem corrigida lado a lado
side_by_side = np.hstack((img1, corrected_image))
cv2.imshow('Original vs. Corrigida', side_by_side)
cv2.imwrite('my_image.png', side_by_side)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('cu', corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()