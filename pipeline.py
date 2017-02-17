import glob
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from line import Line

PROCESS_VIDEO = True
DEBUG = not True
OPTIMIZE = False # Starting value
LEFT_FIT, RIGHT_FIT = Line(), Line() # Current lines
CAL_VARS = None

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(gray, sobel_kernel=3, orient='x', 
                     thresh_min=0, thresh_max=255):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, 
                                          ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, 
                                          ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, 
    # but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) 
                & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, 
    # zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) 
                & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) 
                & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def calibration():

    # First, compute the camera calibration matrix 
    # and distortion coefficients given a set of chessboard images 
    # (in the camera_cal folder in the repository).
    # prepare object points, 
    # like (0,0,0), (1,0,0), (2,0,0), ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and img points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        print("Calibrating camera with image %s" % fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    #cv2.destroyAllWindows()
    cal_vars = cv2.calibrateCamera(objpoints, imgpoints, 
                               gray.shape[::-1],None,None)
    if cal_vars[0]:
        print("Calibration succeeded")
        return cal_vars
    else:
        print("Calibration error (here are cal_vars: %s)" % cal_vars)
        return None
 
# Define a function that thresholds channels of HSV
def hsv_select(img, h_thresh=(0, 255), s_thresh=(0, 255), v_thresh=(0,255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # BGR! NOT RGB!
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(  ((h_channel > h_thresh[0]) & (h_channel <= h_thresh[1]))
                   & ((s_channel > s_thresh[0]) & (s_channel <= s_thresh[1]))
                   & ((v_channel > v_thresh[0]) & (v_channel <= v_thresh[1]))
                  )] = 1 # & or |?
    return binary_output


# Set the width of the windows +/- margin
MARGIN = 25
# Set minimum number of pixels found to recenter window
MINPIX = 10

def visualize(original_img, binary_warped, Minv, 
              text=None):


    # Create an image to draw on and an image to show the selection window
    warp_zero = np.zeros_like(binary_warped) 
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Color in left and right line pixels
    warp_zero[LEFT_FIT.ally , LEFT_FIT.allx]  = [255, 0, 0]
    warp_zero[RIGHT_FIT.ally, RIGHT_FIT.allx] = [0, 0, 255]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([LEFT_FIT.allx, LEFT_FIT.ally]))])
    pts_right = np.array([np.flipud(
                         np.transpose(np.vstack([RIGHT_FIT.allx, RIGHT_FIT.ally])))
                         ])
    pts = np.hstack((pts_left, pts_right))

    # Force Trapezoid (make horizontal points level)
    #if pts_left[0][0]  > pts_right[0][0]:  pts_right[0][0]  = pts_left[0][0] 
    #if pts_left[-1][0] < pts_right[-1][0]: pts_right[-1][0] = pts_left[-1][

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warp_zero, np.int_([pts]), (0,255, 0))
    cv2.polylines(warp_zero, np.int_([pts_left]) , 0, (255,0,0), thickness=30)
    cv2.polylines(warp_zero, np.int_([pts_right]), 0, (255,0,0), thickness=30)
    # Top line and bottom line (horizontal)
    cv2.polylines(warp_zero, np.array([pts_left[-1].astype(np.int32), 
                                      pts_right[-1].astype(np.int32)]),  0, (255,0,0),
                  thickness=20)
    cv2.polylines(warp_zero, np.array([pts_left[0].astype(np.int32) , 
                                      pts_right[0].astype(np.int32)]) ,  0, (255,0,0),
                  thickness=20)

    img_size = warp_zero.shape[0:2]
    img_size = img_size[::-1] # Reverse order
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(warp_zero, #color_warp, #window_img, 
                                  Minv, img_size,
                                  flags=cv2.INTER_LINEAR)

    result = cv2.addWeighted(original_img, 1, newwarp, 
                             0.7, #0.5, #0.3, 
                             0)
    if DEBUG: plt.imshow(result); plt.show()

    if text:
        overlay = result.copy()
        output  = result.copy()
        cv2.putText(overlay, text,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.addWeighted(overlay, 1, output, 0.5, 0, output)

    return output

def sliding_window_search(out_img, binimg, nonzerox, nonzeroy):

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binimg.shape[0]/nwindows)
    # Current positions to be updated for each window
    leftx_current  = LEFT_FIT.get_current_xbase()
    rightx_current = RIGHT_FIT.get_current_xbase()
    print(leftx_current, rightx_current)
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binimg.shape[0] - (window+1)*window_height
        win_y_high = binimg.shape[0] - window*window_height
        win_xleft_low = leftx_current - MARGIN
        win_xleft_high = leftx_current + MARGIN
        win_xright_low = rightx_current - MARGIN
        win_xright_high = rightx_current + MARGIN
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = (((nonzeroy >= win_y_low) 
                         & (nonzeroy < win_y_high) 
                         & (nonzerox >= win_xleft_low) 
                         & (nonzerox < win_xleft_high)).nonzero()[0])
        good_right_inds = (((nonzeroy >= win_y_low) 
                          & (nonzeroy < win_y_high) 
                          & (nonzerox >= win_xright_low) 
                          & (nonzerox < win_xright_high)).nonzero()[0])
        # Append these indices to the lists (If list is non-empty)
        if len(good_left_inds)  > 0:  
            left_lane_inds.append(good_left_inds)
        if len(good_right_inds) > 0:  
            right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > MINPIX:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            LEFT_FIT.set_base_value(leftx_current)
        if len(good_right_inds) > MINPIX:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            RIGHT_FIT.set_base_value(rightx_current)

    return left_lane_inds, right_lane_inds
  
def visualize_sliding_windows(binary_warped, nonzeroy, nonzerox,
                             left_lane_inds, right_lane_inds):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = (LEFT_FIT.current_fit[0]*ploty**2 
               + LEFT_FIT.current_fit[1]*ploty + LEFT_FIT.current_fit[2])
    right_fitx = (RIGHT_FIT.current_fit[0]*ploty**2 
               + RIGHT_FIT.current_fit[1]*ploty + RIGHT_FIT.current_fit[2])

    out_img = np.zeros_like(binary_warped)
    out_img[LEFT_FIT.ally, LEFT_FIT.allx] = 255 #[255, 0, 0]
    out_img[RIGHT_FIT.ally, RIGHT_FIT.allx] = 100 #[0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

def optimized_search(binary_warped, nonzerox, nonzeroy):

    global LEFT_FIT, RIGHT_FIT
    # Assume you now have a new warped binary image 
    # from the next frame of video
    # It's now much easier to find line pixels!

    l_fit = LEFT_FIT.current_fit
    r_fit = RIGHT_FIT.current_fit
    # Return these as 1-item lists
    # (for consistency with the ret value of sliding window function)
    left_lane_inds  = \
        [(((nonzerox > (l_fit[0]*(nonzeroy**2) 
                      + l_fit[1]*nonzeroy 
                      + l_fit[2] - MARGIN)) 
         & (nonzerox < (l_fit[0]*(nonzeroy**2) 
                      + l_fit[1]*nonzeroy 
                      + l_fit[2] + MARGIN))).nonzero()[0])]
    right_lane_inds = \
        [(((nonzerox > (r_fit[0]*(nonzeroy**2) 
                      + r_fit[1]*nonzeroy 
                      + r_fit[2] - MARGIN)) 
         & (nonzerox < (r_fit[0]*(nonzeroy**2) 
                      + r_fit[1]*nonzeroy 
                      + r_fit[2] + MARGIN))).nonzero()[0])]

    return left_lane_inds, right_lane_inds


YM_PER_PIX = 30.0/720.0
XM_PER_PIX = 3.7/700.0
def compute_line_metrics(combined, img_size):
    """
    This function updates many of the variables that are a part
    of the Line class
    """

    # Generate x and y values for plotting
    # We use the averaged coefficients based on a last n frames
    fity = np.linspace(0, combined.shape[0]-1, combined.shape[0] )
    fit_leftx = (LEFT_FIT.best_fit[0]*fity**2 
                 + LEFT_FIT.best_fit[1]*fity + LEFT_FIT.best_fit[2])
    fit_rightx = (RIGHT_FIT.best_fit[0]*fity**2 
                 + RIGHT_FIT.best_fit[1]*fity + RIGHT_FIT.best_fit[2])

   
    ploty = np.linspace(0, combined.shape[0]-1, combined.shape[0])
    # to cover same y-range as image
    y_eval = np.max(ploty)
    LEFT_FIT.roc_pix  = \
            ((1 + (2*LEFT_FIT.best_fit[0]*y_eval
                   + LEFT_FIT.best_fit[1])**2)**1.5) \
            / np.absolute(2*LEFT_FIT.best_fit[0])
    RIGHT_FIT.roc_pix = \
            ((1 + (2*RIGHT_FIT.best_fit[0]*y_eval
                   + RIGHT_FIT.best_fit[1])**2)**1.5) \
            / np.absolute(2*RIGHT_FIT.best_fit[0])

    # Fit new polynomials to x,y in world space
    #left_fit_cr  = np.polyfit(LEFT_FIT.ally*YM_PER_PIX,  LEFT_FIT.allx*XM_PER_PIX,  2)
    #right_fit_cr = np.polyfit(RIGHT_FIT.ally*YM_PER_PIX, RIGHT_FIT.allx*XM_PER_PIX, 2)
    # Calculate the new radius of curvature
    l_curr_m_fit = LEFT_FIT.current_metric_fit
    r_curr_m_fit = RIGHT_FIT.current_metric_fit
    LEFT_FIT.roc_metric = \
                    ((1 + (2*l_curr_m_fit[0]*y_eval*YM_PER_PIX
                           + l_curr_m_fit[1])**2)**1.5) \
                    / np.absolute(2*l_curr_m_fit[0])
    RIGHT_FIT.roc_metric = \
                    ((1 + (2*r_curr_m_fit[0]*y_eval*YM_PER_PIX
                           + r_curr_m_fit[1])**2)**1.5) \
                    / np.absolute(2*r_curr_m_fit[0])
    #epsilon = 10 # The two lines have "fused"
    if True: #(LEFT_FIT.roc_metric  < 100.0 
    # or RIGHT_FIT.roc_metric < 100.0
    # or abs(LEFT_FIT.roc_metric - RIGHT_FIT.roc_metric) < epsilon):
        LEFT_FIT.detected  = True
        RIGHT_FIT.detected = True
        # If the ROC of a lane line falls below a certain value 
        # it is clearly invalid and we need to go
        # back to the sliding window temporarily
    LEFT_FIT.line_base_pos  = \
            abs(LEFT_FIT.best_xbase-img_size[0]/4)*XM_PER_PIX
    RIGHT_FIT.line_base_pos = \
            abs(RIGHT_FIT.best_xbase-img_size[0]*3/4)*XM_PER_PIX


def pipeline(src_img):

    global OPTIMIZE, LEFT_FIT, RIGHT_FIT
    ret, mtx, dist, rvecs, tvecs = CAL_VARS

    original_image = cv2.undistort(src_img, mtx, dist, None, mtx)
    if DEBUG: cv2.imwrite('./output_images/undistort_output.png', 
                          original_image)
    img_size = original_image.shape[0:2]
    img_size = img_size[::-1] # Reverse order
    
    # -50 = hood of car
    src = np.float32([#[(img_size[0]/2)-80,   img_size[1]/2+100],
                       [(img_size[0]/2)-100, img_size[1]*5/8], # better for challenge
                       [(img_size[0]/6)-20,   img_size[1]-50], 
                       [(img_size[0]*5/6)+20, img_size[1]-50],
                       [(img_size[0]/2)+100, img_size[1]*5/8] # better for challenge
                      #[(img_size[0]/2)+80,   img_size[1]/2+100]]
                     ])
    dst = np.float32([[(img_size[0]/4), 0],
                      [(img_size[0]/4), img_size[1]],
                      [(img_size[0]*3/4), img_size[1]],
                      [(img_size[0]*3/4), 0]])

    # One especially smart way to do this would be to use 
    # four well-chosen corners that were automatically detected 
    # during the undistortion steps
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_cimage = \
        cv2.warpPerspective(original_image, M, img_size,
                                       flags=cv2.INTER_LINEAR)
    if DEBUG: 
        cv2.imwrite('./output_images/warped_straight_lines.jpg',
                    warped_cimage)
        #plt.imshow(warped_cimage)
        #plt.show()

    # Choose a Sobel kernel size
    ksize = 3 
    # Choose an odd number >= 3 to smooth gradient measurements
    #gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(warped_cimage, cv2.COLOR_BGR2GRAY)

    # Apply each of the thresholding functions
    dir_binary = dir_threshold(gray, sobel_kernel=15,
                               thresh=(0.7, 1.3))
    mag_binary = mag_thresh(   gray, sobel_kernel=3,
                               mag_thresh=(50, 100))
 
    #hls_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HLS)
    hls_image = cv2.cvtColor(warped_cimage, cv2.COLOR_BGR2HLS)
     
    #gradx_s = abs_sobel_thresh(hls_image[:,:,2], 
    #                           orient='x', 
    #                           sobel_kernel=ksize,
    #                           thresh_min=50, thresh_max=200)
    gradx_l = abs_sobel_thresh(hls_image[:,:,1], 
                               orient='x',
                               sobel_kernel=ksize,
                               thresh_min=50, thresh_max=200)
    #gradx = np.zeros_like(dir_binary)
    #gradx[(gradx_l == 1) | (gradx_s == 1)] = 1
    gradx = gradx_l     

    #grady_s = abs_sobel_thresh(hls_image[:,:,2], 
    #                           orient='y', 
    #                           sobel_kernel=ksize, 
    #                           thresh_min=50, thresh_max=200)
    grady_l = abs_sobel_thresh(hls_image[:,:,2], orient='y',
                               sobel_kernel=ksize,
                               thresh_min=50, thresh_max=200)
    #grady = np.zeros_like(dir_binary)
    #grady[(grady_l == 1) | (grady_s == 1)] = 1
    grady = grady_l 

    hsv_binary_y = hsv_select(warped_cimage, 
          h_thresh=(0,50), s_thresh=(100,255), v_thresh=(100,255))
    hsv_binary_w = hsv_select(warped_cimage,
          h_thresh=(20,255), s_thresh=(0,80), v_thresh=(180, 255))
    hsv_binary = np.zeros_like(dir_binary)
    hsv_binary[(hsv_binary_y == 1) | (hsv_binary_w == 1)] = 1
    #if DEBUG: print('hsv'); plt.imshow(hsv_binary); plt.show()

    combined = np.zeros_like(dir_binary)
    combined[(((gradx == 1) | (grady == 1)) 
            | ((mag_binary == 1) & (dir_binary == 1)))
            | (hsv_binary == 1)
             ] = 1
    if DEBUG: 
        print('combined'); plt.imshow(combined); plt.show()
        cv2.imwrite('./output_images/binary_combo_example.jpg',
                    combined)
    #combined = cv2.warpPerspective(combined, M, img_size,
    #                               flags=cv2.INTER_LINEAR)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = combined.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    histogram = np.sum(combined[combined.shape[0]/2:,:], axis=0)
    #if DEBUG: plt.plot(histogram); plt.show()
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((combined, combined, combined))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    LEFT_FIT.set_base_value(leftx_base)
    RIGHT_FIT.set_base_value(rightx_base)

    left_lane_inds  = []
    right_lane_inds = []
    if LEFT_FIT.detected and RIGHT_FIT.detected:
        left_lane_inds, right_lane_inds = optimized_search(combined, 
                                                   nonzerox, nonzeroy)

    # If detected below, we will set them back to true
    LEFT_FIT.detected = False
    RIGHT_FIT.detected = False
    if not left_lane_inds or not right_lane_inds:
        print("Resorting to sliding window")
        left_lane_inds, right_lane_inds = \
                sliding_window_search(out_img, combined, 
                                      nonzerox, nonzeroy)
        #if not left_lane_inds or not right_lane_inds:

    if left_lane_inds and right_lane_inds:
        # Concatenate the arrays of indices
        LEFT_FIT.lane_inds  = np.concatenate(left_lane_inds)
        RIGHT_FIT.lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        LEFT_FIT.allx  = nonzerox[LEFT_FIT.lane_inds]
        LEFT_FIT.ally  = nonzeroy[LEFT_FIT.lane_inds]
        RIGHT_FIT.allx = nonzerox[RIGHT_FIT.lane_inds]
        RIGHT_FIT.ally = nonzeroy[RIGHT_FIT.lane_inds]
   
        new_left  = False; 
        new_right = False
        # Fit a second order polynomial to each
        if len(LEFT_FIT.ally ) > 0 and len(LEFT_FIT.allx ) > 0:
            new_left_fit = np.polyfit(LEFT_FIT.ally,  LEFT_FIT.allx,  2)
            # Fit new polynomial to x,y in world space
            new_left_fit_m  = np.polyfit(LEFT_FIT.ally*YM_PER_PIX,  
                                         LEFT_FIT.allx*XM_PER_PIX, 2)
            LEFT_FIT.set_current_fit(new_left_fit)
            LEFT_FIT.set_current_metric_fit(new_left_fit_m)
            LEFT_FIT.diffs  = abs(LEFT_FIT.current_fit - new_left_fit)
            if LEFT_FIT.diffs[0] < 0.1 and LEFT_FIT.diffs[1] < 0.1: 
                new_left = True # Sanity check

        if len(RIGHT_FIT.ally) > 0 and len(RIGHT_FIT.allx) > 0:
            new_right_fit = np.polyfit(RIGHT_FIT.ally, RIGHT_FIT.allx, 2)
            # Fit new polynomial to x,y in world space
            new_right_fit_m  = np.polyfit(RIGHT_FIT.ally*YM_PER_PIX, 
                                          RIGHT_FIT.allx*XM_PER_PIX,  2)
            RIGHT_FIT.set_current_fit(new_right_fit)
            RIGHT_FIT.set_current_metric_fit(new_right_fit_m)
            RIGHT_FIT.diffs = abs(RIGHT_FIT.current_fit - new_right_fit)
            if RIGHT_FIT.diffs[0] < 0.1 and RIGHT_FIT.diffs[1] < 0.1:
                new_right = True # Sanity check

        if new_left and new_right:
            if ((new_left_fit[0] > 0 and new_right_fit[0] > 0) 
               or (new_left_fit[1] < 0 and new_right_fit[1] < 0)):
                LEFT_FIT.detected = new_left
                RIGHT_FIT.detected = new_right
        else:
            LEFT_FIT.detected  = False
            RIGHT_FIT.detected = False

        if new_left and new_right:
            compute_line_metrics(combined, img_size)   

    off_cen = (LEFT_FIT.line_base_pos
             + RIGHT_FIT.line_base_pos)/2

    # Now our radius of curvature is in meters
    roc_text = ("L_ROC=%sm R_ROC=%sm off_cen=%sm" 
               % (LEFT_FIT.roc_metric, 
                  RIGHT_FIT.roc_metric, 
                  off_cen))
    print(roc_text)
    # Example values: 632.1 m 626.2 m 0.0
    if DEBUG: visualize_sliding_windows(combined, nonzeroy, nonzerox,
                                     left_lane_inds, right_lane_inds)

    # combined or warped_image_color?
    result = visualize(original_image, warped_cimage, Minv,
                       text=roc_text)

    if DEBUG:
        cv2.imwrite('./output_images/example_output.jpg', result)
        plt.imshow(result); plt.show()

    return result 

    

def main():
    global CAL_VARS

    # So we don't have to calibrate every time
    try:
        with open('./cal_vars.p', 'rb') as _input:
            CAL_VARS = pickle.load(_input)
        print('Successfully loaded prior calibration variables')
    except Exception as e:
        print('Got exception %s when trying to load calibration variables' % e)
        print('No saved calibration variables, calibrating now (saving for future use)')
        CAL_VARS = calibration()
        with open('./cal_vars.p', 'wb') as output:
            pickle.dump(CAL_VARS, output)    
        
 
    # TODO: For a series of test images (in the test_images folder in the repository): 
    if CAL_VARS:
        images = glob.glob('test_images/*.jpg')
        print("Looking at images %s" % images)
        for idx, fname in enumerate(images):
            image = cv2.imread(fname)
            result = pipeline(image)
            out_fn = './output_images/%s' % fname.split('/')[-1]
            print('Saving output to %s' % out_fn)
            cv2.imwrite(out_fn, result)
        if DEBUG: return
        # NOTE: save example images from each stage of your pipeline to the output_images folder 
        #       and provide a description of what each image is in your README for the project.

        # Run your algorithm on a video. In the case of the video, you must search for the lane lines in the first few frames, 
        # and, once you have a high-confidence detection, use that information to track the position and curvature of the lines 
        # from frame to frame. Save your output video and include it with the submission.
  
        output_file = 'project_video_out.mp4'
        clip = VideoFileClip('project_video.mp4')
        out_clip = clip.fl_image(pipeline)
        out_clip.write_videofile(output_file, audio=False)
        
        output_file = 'challenge_video_out.mp4'
        clip = VideoFileClip('challenge_video.mp4')
        out_clip = clip.fl_image(pipeline)
        out_clip.write_videofile(output_file, audio=False)

        output_file = 'harder_challenge_video_out.mp4'
        clip = VideoFileClip('harder_challenge_video.mp4')
        out_clip = clip.fl_image(pipeline)
        out_clip.write_videofile(output_file, audio=False)

if __name__ == "__main__":
    main()
