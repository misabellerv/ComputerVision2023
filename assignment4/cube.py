import numpy as np
import cv2


cube_vertices = np.array([
    [2, 2, 2],
    [-2, 2, 2],
    [-2, 2, -2],
    [2, 2, -2],
    [2, -2, 2],
    [-2, -2, 2],
    [-2, -2, -2],
    [2, -2, -2]
])

image_points = np.array([
    [422, 323],
    [178, 323],
    [118, 483],
    [482, 483],
    [438, 73],
    [162, 73],
    [78, 117],
    [522, 117]
])


# image width and height
image_width = 700
image_height = 500

image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

edges = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

for edge in edges:
    point1 = tuple(image_points[edge[0]].astype(int))
    point2 = tuple(image_points[edge[1]].astype(int))
    cv2.line(image, point1, point2, (0, 255, 0), 1)

cv2.imshow('Cube Projection', image)
cv2.imwrite('cube.jpg',image )
cv2.waitKey(0)
cv2.destroyAllWindows()