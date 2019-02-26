## Advance Lane Finding



[//]: # (Image References)

[image1]: ./output_images/calibration18.jpg "Distorted chessboard"
[image2]: ./output_images/undistorted_calibration18.jpg "Undistorted chessboard"
[image3]: ./output_images/undistorted_image2.jpg "Undistorted straight line"
[image4]: ./output_images/thresholded2.jpg "Binary Image"
[image5]: ./output_images/warped_bird_view.jpg "Warp Example"
[image6]: ./output_images/poly_fit_2.jpg "Fit Visual"
[image7]: ./output_images/fittingpoly_findinglanes.jpg "Output"
[video1]: ./output_images/.mp4 "Video"
[video2]: ./output_images/.mp4 "challenge Video"


### Camera Calibration

#### 1 Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

The first step of the project is to do the camera calibration. It is necessary because the lenses of the camera distort the image of the real world.

One of the ways to address this distortion is to calibrate our camera if we know how much the lenses distort our image. 

To start with, we can take a look at a chessboard image before we calibrate and undistort the image.
This will allow us to calculate how much distortion is present and correct for it (most of the heavy lifting is done by OpenCV).
We can do this same calibration with a set of chessboard images. 

We can use the OpenCV package to find these intersection points using the findChessboardCorners function which return a set of detected corner coordinates if the status is 1 or no corners if no corners could be found (status of 0).

we can use opencv function cv2.calibrateCamera() to find the arrays that describe the distortion of the lenses.

The calibration step uses these object (actual flat chessboard) and image (distorted image) points to calculate a camera matrix, distortion matrix, and a couple other data points related to the position of the camera.

The cv2.undistort() OpenCV package can be applied to undistort the image by using the camera Matrix and the distortionCoeffs as the arguments to cv2.undistort() along with the image which returns an undistorted iamge.

Example of distorted chessboard:

![alt text][image1] 


Example of undistorted  chessboard:

![alt text][image2] 


Now applying the same to the original image in the folder test_images:

The undistorted Straight line:

![alt text][image3] 



### Use color transforms, gradients, etc., to create a thresholded binary image.

we have undistorted images now so we can start to detect lane lines in the images. 

We would like to calculate the curvature of the lanes to decide how to steer our car.

Here we use different color spaces, gradient, and direction thresholding techniques to extract lane lines.

#### Sobel Filter Thresholding:-
The sobel filter takes the derivative in the x or y direction to highlight pixels where there is a large change in the pixel values.

The sobel filter can be applied with OpenCV package cv2.Sobel().

#### Magnitude and Direction of Gradient Thresholding:-

To potentially clean up the basic sobel filter thresholding is to use the magnitude of the x and y gradients to filter out gradients that are below a certain threshold. 

#### Color Thresholding:

We do a color threshold filter to pick only yellow and white elements, using opencv convert color to HSV space (Hue, Saturation and Value). The HSV dimension is suitable to do this, because it isolates color (hue), amount of color (saturation) and brightness (value). We define the range of yellow independent on the brightness .

The Binary Thresholded image :

![alt text][image4] 


### perspective transform.

The perspective transform let’s us view a lane from above, this will be useful for calculating the lane curvature.

Opencv provide two functions getPerspectiveTransform and warpPerspective to perform this task.

The opencv function getPerspectiveTransform needs 4 origins and destinations points. The origins are like a trapezium containing the lane. The destination is a rectangle.

After manually examining a sample image, I extracted the vertices to perform a perspective transform.
The polygon with these vertices is drawn on the image for visualization. Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.

![alt text][image5] 

### Detect lane pixels and fit to find the lane boundary.

We now have a thresholded warped image and we’re ready to map out the lane lines! There are many ways we could go about this, but using peaks in the histogram works well.

After applying calibration, thresholding, and a perspective transform to a road image, we should have a binary image where the lane lines stand out clearly. However, we still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

Take a histogram across the bottom of the image.
Find the histogram peaks to identify the lane lines at the bottom of the image.
Divide the image into a vertical stack of narrow horizontal slices.
Select activated pixels (remember, the input is a binary image) only in a "neighborhood" of our current estimate of the lane position. This neighborhood is the "sliding window." To bootstrap the process, our initial estimate of the lane line location is taken from the histogram peak steps listed above. Essentially, we are removing "outliers"
Estimate the new lane-line location for this window from the mean of the pixels falling within the sliding window.
March vertically up through the stack, repeating this process.
Select all activated pixels within all of our sliding windows.

![alt text][image6] 


### Determine the curvature of the lane and vehicle position with respect to center.

Self-driving cars need to be told the correct steering angle to turn, left or right.

We can calculate this angle if we know a few things about the speed and dynamics of the car and how much the lane is curving.

One way to calculate the curvature of a lane line, is to fit a 2nd degree polynomial to that line, and from this we can easily extract useful information.

For a lane line that is close to vertical, we can fit a line using this formula: f(y) = Ay^2 + By + C, where A, B, and C are coefficients. 

A gives us the curvature of the lane line, 
B gives us the heading or direction that the line is pointing, and 
C gives us the position of the line based on how far away it is from the very left of an image (y = 0).


![alt text][image7] 

### The Video

The pipeline is applied to a video

Here's a [link to my video result](./output_images/project_video_output.mp4)


### The Challenge Video

The pipeline is applied to the challenge video also but there are some more improvements to be done as we see lot of failures when the care is taking a turn and when the car is passing a tunnel.

Here's a [link to my video result](./output_images/challenge_video_output1.mp4)

### References:

https://medium.com/@edvoas/advanced-lane-finding-a4bb8356824d

https://medium.com/intro-to-artificial-intelligence/self-driving-car-nanodegree-advanced-lane-finding-9c806b277a31

https://medium.com/@zhuangh/advanced-lane-finding-project-d6c64f487564
