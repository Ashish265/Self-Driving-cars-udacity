## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---

The goals / steps of this project are the following:

1 - Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

The first step of the project is to do the camera calibration. It is necessary because the lenses of the camera distort the image of the real world.

One of the ways to address this distortion is to calibrate our camera if we know how much the lenses distort our image. 

To start with, we can take a look at a chessboard image before we calibrate and undistort the image.
This will allow us to calculate how much distortion is present and correct for it (most of the heavy lifting is done by OpenCV).
We can do this same calibration with a set of chessboard images. 

We can use the OpenCV package to find these intersection points using the findChessboardCorners function which return a set of detected corner coordinates if the status is 1 or no corners if no corners could be found (status of 0).

we can use opencv function cv2.calibrateCamera() to find the arrays that describe the distortion of the lenses.

The calibration step uses these object (actual flat chessboard) and image (distorted image) points to calculate a camera matrix, distortion matrix, and a couple other data points related to the position of the camera.

2 -  Apply a distortion correction to raw images.

The cv2.undistort() OpenCV package can be applied to undistort the image by using the camera Matrix and the distortionCoeffs as the arguments to cv2.undistort() along with the image which returns an undistorted iamge.


3- Use color transforms, gradients, etc., to create a thresholded binary image.

we have undistorted images now so we can start to detect lane lines in the images. 

We would like to calculate the curvature of the lanes to decide how to steer our car.

Here we use different color spaces, gradient, and direction thresholding techniques to extract lane lines.

Sobel Filter Thresholding:-
The sobel filter takes the derivative in the x or y direction to highlight pixels where there is a large change in the pixel values.

The sobel filter can be applied with OpenCV package cv2.Sobel().

Magnitude and Direction of Gradient Thresholding:-

To potentially clean up the basic sobel filter thresholding is to use the magnitude of the x and y gradients to filter out gradients that are below a certain threshold. 

Color Thresholding:

We do a color threshold filter to pick only yellow and white elements, using opencv convert color to HSV space (Hue, Saturation and Value). The HSV dimension is suitable to do this, because it isolates color (hue), amount of color (saturation) and brightness (value). We define the range of yellow independent on the brightness .

To find the contrast, we use the Sobel operator. It is an derivative, under the hood. If the difference in color between two points is very high, the derivative will be high.

4 - Apply a perspective transform to rectify binary image ("birds-eye view").

A perspective transform maps the points in a given image to different, desired, image points with a new perspective. 

The perspective transform let’s us view a lane from above, this will be useful for calculating the lane curvature.

Opencv provide two functions getPerspectiveTransform and warpPerspective to perform this task.

The opencv function getPerspectiveTransform needs 4 origins and destinations points. The origins are like a trapezium containing the lane. The destination is a rectangle.


5 - Detect lane pixels and fit to find the lane boundary.

We now have a thresholded warped image and we’re ready to map out the lane lines! There are many ways we could go about this, but using peaks in the histogram works well.

After applying calibration, thresholding, and a perspective transform to a road image, we should have a binary image where the lane lines stand out clearly. However, we still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

Take a histogram across the bottom of the image.
Find the histogram peaks to identify the lane lines at the bottom of the image.
Divide the image into a vertical stack of narrow horizontal slices.
Select activated pixels (remember, the input is a binary image) only in a "neighborhood" of our current estimate of the lane position. This neighborhood is the "sliding window." To bootstrap the process, our initial estimate of the lane line location is taken from the histogram peak steps listed above. Essentially, we are removing "outliers"
Estimate the new lane-line location for this window from the mean of the pixels falling within the sliding window.
March vertically up through the stack, repeating this process.
Select all activated pixels within all of our sliding windows.
Fit a quadratic function to these selected pixels, obtaining model parameters.

6 - Determine the curvature of the lane and vehicle position with respect to center.

Self-driving cars need to be told the correct steering angle to turn, left or right.

We can calculate this angle if we know a few things about the speed and dynamics of the car and how much the lane is curving.

One way to calculate the curvature of a lane line, is to fit a 2nd degree polynomial to that line, and from this we can easily extract useful information.

For a lane line that is close to vertical, we can fit a line using this formula: f(y) = Ay^2 + By + C, where A, B, and C are coefficients. 

A gives us the curvature of the lane line, 
B gives us the heading or direction that the line is pointing, and 
C gives us the position of the line based on how far away it is from the very left of an image (y = 0).



References::

https://towardsdatascience.com/robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3

http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/

https://medium.com/@edvoas/advanced-lane-finding-a4bb8356824d

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  




To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

