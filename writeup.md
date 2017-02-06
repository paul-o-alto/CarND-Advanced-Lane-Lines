##Advanced Lane Finding Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project were the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
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
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README  

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function 'calibration' in "./pipeline.py" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. An example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Color transforms, gradients or other methods to create a thresholded binary image. 
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `pipeline.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Perspective transform and an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `pipeline.py` (output_images/examples/example.py).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

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

####4. Lane-line pixels and fitting their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #1 through #10 in my code in `pipeline.py`

####6. An example image of the result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #1 through #10 in my code in `pipeline.py` in the function `pipeline()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. A link to the final video output.

Here's a [link to my video result](./project_video_out.mp4)

---

###Discussion

####1. Problems / issues I faced in the implementation of this project.  Where will the pipeline likely fail?  What could be done to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I FACED problems with getting my pipeline to work reliably on the challenge videos. I decided to incorporate the Line class into my code in an effort to achieve success in those more challenging cases. This led my original output video to be more jittery. I believe that overall the added smoothing from this abstraction will result in better performance on all videos when some of the little regressions in performance are fixed.

