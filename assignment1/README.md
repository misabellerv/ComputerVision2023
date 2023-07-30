# Assignment 1 - Projections, Homographies, and Perspective Correction 🏠🔍🌇

Welcome to Assignment 1 of the ComputerVision2023 repository! In this assignment, we'll explore three key topics: **Projections**, **Homographies**, and **Perspective Correction**. Let's delve into each section:

## 1. Projections (1️⃣)

In this section, we focus on the mapping of 3D points to the 2D projective space using a 3x4 projection matrix. The algorithm involves the following tasks:

### 1.1 Projection Matrix Estimation 🔍

We estimate the 3x4 projection matrix `P` using Singular Value Decomposition (SVD) of the system `Ap = 0`, where `p` is a column vector containing the entries of the main projection matrix. Once we find `P`, we can use it to project 3D points to 2D and check if these points match the annotated 2D points. The projection error, i.e., the distance between the projected and annotated points, is computed to evaluate the accuracy of the mapping. The code for this task can be found in `projection_matrix.py`.

### 1.2 Adding White Noise 📝

Next, we add white noise to all 3D points and reevaluate the projection error. This task helps us understand the robustness of the projection method when dealing with noisy data. The code for this task can be found in `white_noise.py`.

### 1.3 Introducing Random Noise 🎲

In the final task of this section, we replace 20% of the 3D points with values in the range [-max3D, +max3D], where `max3D` is the maximum absolute value of the 3D coordinates. This simulates the presence of outliers in the data. The code for this task can be found in `random_noise.py`.

## 2. Homographies (2️⃣)

In this section, we deal with finding homographies between two images of the same scene with slightly different perspectives. The tasks in this section include:

### 2.1 Homography Estimation 🔍

We annotate corresponding points in both images and then compute the homography matrix `H` that maps the homogeneous coordinates (x, y, 1) of one image to the coordinates of the other. This allows us to find the transformation between the two images. The code for this task can be found in `homography.py`.

## 3. Perspective Correction (3️⃣)

In this section, we apply the concepts of homography to correct perspective distortions in images. The process involves the following steps:

### 3.1 Measuring Reference Distances 📏

We start by measuring the base and height between three frames in the figure. This information helps us establish a coordinate system that includes the front view of the frames with the determined points.

### 3.2 Homography Matrix Calculation 🔍

Next, we select four non-collinear points in the distorted images, which correspond to the points determined by the reference distances. We then calculate the homography matrix for these points. Applying the homography matrix to the distorted image results in a perspective-corrected image. The code for this task can be found in `perspective_correction.py`.

## Let's Connect! 🤝

If you have any questions or need further assistance with the code or the concepts involved, feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/misabellerv). Enjoy exploring projections, homographies, and perspective correction in Computer Vision!

Happy coding! 📐🔍🌇🚀
