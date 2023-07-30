# Assignment 0 - Image Rectification using Two-Step Method ✨

Welcome to Assignment 0 of the ComputerVision2023 repository! In this assignment, we will be focusing on **image rectification** using the Two-Step Method. The Two-Step Method is a powerful technique that allows us to remove both **projective distortions** and **affine distortions**, resulting in metric rectification.

## Two-Step Method Overview 📝

The Two-Step Method is a common approach to rectify distorted images. It involves two main steps:

1. **Projective Distortion Removal**: In this step, we address projective distortions by estimating the homography that maps the distorted image to a canonical view. The homography is then used to rectify the image and remove projective distortions.

2. **Affine Distortion Removal**: Once the projective distortions are removed, we move on to address affine distortions. Affine transformations are applied to the projective rectified image to remove any remaining distortions, achieving metric rectification.

## Getting Started 🚀

To start rectifying an image with projective distortion, follow these steps:

1. Open the `projective_distortion.py` script.
2. Replace `your_image_path.jpg` with the path to your input image that contains projective distortion.
3. Run the script. This will generate a new image named "AffineRectifiedImage.jpg" that is rectified for projective distortions.

After obtaining the "AffineRectifiedImage.jpg," proceed with the next step to achieve metric rectification:

1. Open the `affine_distortion.py` script.
2. Replace `AffineRectifiedImage.jpg` with the path to the image generated in the previous step.
3. Run the script. This will perform affine distortion removal on the rectified image and output the final metrically rectified image.

## Note ⚠️

Ensure that you have the necessary libraries and dependencies installed to run the scripts successfully. If any issues arise, please check the documentation and requirements of the specific libraries being used in the code.

## Example Results 📷

The final output image obtained after the Two-Step Method will exhibit significantly reduced distortions, and lines that are physically parallel and perpendicular in the real world will appear as such in the rectified image.

## Let's Connect! 🤝

If you have any questions or need further assistance with the code or the concepts involved, feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/misabellerv). I'm excited to see your metrically rectified images!

Happy rectifying! 📐🌟
