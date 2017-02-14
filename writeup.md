##Advanced Lane Finding Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project were the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binary_combo_example.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/example_output.jpg "Output"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README  

###Camera Calibration

####1. Computing the camera matrix and distortion coefficients. (example of distortion correction included)

The code for this step is contained in the function `calibration` in `./pipeline.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. An example of a distortion-corrected image.
To demonstrate this step, I applied the distortion correction to one of the test images:
![alt text][image2]
####2. Color transforms, gradients or other methods to create a thresholded binary image. 
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps are contained in the `pipeline` function in `pipeline.py`).  I used gradients in the x and y direction ORed them with one another. I also used combined gradients in both the x and y direction and thresholded them based on their absolute magnitude and direction. Color thresholding was done in HSV color space and use 3 seperate thresholds for each channel, respectively.

![alt text][image3]

####3. Perspective transform and an example of a transformed image.

The code for my perspective transform includes a function called `cv2.warpPerspective()`, which appears in the first part of the `pipeline` function `pipeline.py`.  The `cv2.warpPerspective()` function takes as inputs an image (`img`), as well a perspective transform matrix `M` and the size of the image. The matrix `M` was computed using `cv2.getPerspectiveTransform()` and the source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner:

```
src = np.float32([[(img_size[0]/2)-80,   img_size[1]/2+100],
                      [(img_size[0]/6)+10,   img_size[1]],
                      [(img_size[0]*5/6)-10, img_size[1]],
                      [(img_size[0]/2)+80,   img_size[1]/2+100]])
dst = np.float32([[(img_size[0]/4), 0],
                      [(img_size[0]/4), img_size[1]],
                      [(img_size[0]*3/4), img_size[1]],
                      [(img_size[0]*3/4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Lane-line pixels and fitting their positions with a polynomial.

I then proceeded to take the thresholded warped binary image computed before and computed a histogram based on the pixel values. The two spikes in this histogram represent the area where the search for the lane lines begins. A sliding window technique was then used to search for nonzero pixels. These pixels were then used to produce a second-order polynomial `f(y)`. It was necessary to make this function be in terms of `y` because of the fact that the lane lines can become completely vertical.

The sliding window technique was not necessary for every frame. If it had already been performed in a previous frame, you could confine the search closer to where the existing polynomial was plotted. 

![alt text][image5]

####5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.

The calculation of the radius of curvature was done in the function `compute_line_metrics` in `pipeline.py`. A conversion was done to bring the value from pixel space, into world space (in meters). 

The position of the vehicle with respect to center was calculated in `compute_line_metrics` in `pipeline.py`. This was computed using the measurement of the base position of the lane lines and comparing it to where the lines should be (eye-balling an image, they should be at about 1/4 of the width and 3/4 of the width of the image). We measure the difference of the line base positions to these static positions in the image. From this calculation, the results came out to be quite reasonable (~0.3 meters most of the time)

####6. An example image of the result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `pipeline.py` in the function `visualize()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. A link to the final video output.

Here's a [link to my video result](./project_video_out.mp4)

---

###Discussion

####1. Problems / issues I faced in the implementation of this project.  Where will the pipeline likely fail?  What could be done to make it more robust?

I faced problems with getting my pipeline to work reliably on the challenge videos. I decided to incorporate the Line class into my code in an effort to achieve success in those more challenging cases. This led my original output video to be more jittery. I had to extensively refactor my code before I finally got back my previous level of performance on the test images. Unfortunately, the smoothing provided by this Line class did not yield the expected better performance on the challenge videos. More work is clearly needed here.

The pipeline I created will likely fail on images with much more diverse lighting conditions (for example, in the harder challenge video). Also, very steep or irregular road surfaces would probably pose a challenge to the thresholding step of my pipeline and yield a warped binary image that cannot produce a reasonable polynomial.

This implementation could be made more robust by perhaps incorporating various combinations of thresholding steps and utilizing the best one. Also, a system for anomaly rejection (polynomial-wise) could maybe be useful. Perhaps one could use one polynomial over several frames of no lane detection.

