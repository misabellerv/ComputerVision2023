# Assignment 2 - Computer Vision 📸

👋 Welcome to my Assignment 2 GitHub repository for Computer Vision! In this project, we dive into two exciting challenges: Harris Corner Detection and Feature Matching. Let's take a look at the two parts of this assignment:

## Part 1: Harris Corner Detection ✨

In the first part, we built our very own Harris Corner Detector from scratch! 🏗️ You can find the source code for this implementation in the `harris_detector.py` file. We delved into the algorithm details, calculating the covariance matrix using derivatives in x and y for each pixel, applying the Harris responses using the formula R = det(C) - 0.04 * (Tr(C))^2, and finally identifying the most important corners in an image. 🧭

To see the Harris Corner Detector in action, check out the `test_harris.ipynb` file. There, we tested the detector with various images, visualized the detected corners, and explored how it handles different scenarios. 🕵️‍♂️

## Part 2: Feature Matching 🔍

In the second part of the assignment, we tackled another thrilling challenge - Feature Matching! 💫 The implementation for this task can be found in the `feature_matching.py` file. Here, we used the features detected by the Harris Corner Detector to find correspondences between different images. The goal is to identify similar key points in multiple images, allowing us to find corresponding objects or scenes across them. 🌐

## How to Run 🚀

Each part of the assignment has its own source code and test notebook. To run the code, make sure you have the requirements installed, and then simply execute the associated Python notebooks or scripts. Don't forget to add your own images to see how the Harris Corner Detector and Feature Matching perform in your specific scenarios! 🖼️

Have fun exploring the fascinating possibilities of Computer Vision! 🌟

# Requirements and Dependencies 🛠️

- Python 3.x
- Libraries: OpenCV, Numpy, Jupyter, etc.

# Feedback and Contributions 🤝

If you have any questions, feedback, or suggestions, feel free to open an issue or submit a pull request. Your contribution is highly appreciated and helps improve this assignment for everyone! 🙌

Enjoy your time here and happy studying! 📚✨

## Acknowledgements 😊

Special thanks to all involved in the development of this assignment and to you for checking out this project! Together, let's explore and unravel the amazing world of Computer Vision! 🌐👀
