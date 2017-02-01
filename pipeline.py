import glob
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

TRY_CHALLENGE = not False
DEBUG = not True
OPTIMIZE = False # Starting value
LEFT_FIT, RIGHT_FIT = None, None # Current lines
CAL_VARS = None

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(gray, sobel_kernel=3, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def calibration():

    # First, compute the camera calibration matrix and distortion coefficients given a set of chessboard images 
    # (in the camera_cal folder in the repository).
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
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
        ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    cv2.destroyAllWindows()
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
 
# Define a function that thresholds the S-channel of HLS
def hsv_select(img, h_thresh=(0, 180), v_thresh=(0,255), s_thresh=(0,255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[((h_channel > h_thresh[0]) & (h_channel <= h_thresh[1]))
                & ((v_channel > v_thresh[0]) & (v_channel <= v_thresh[1])) 
                & ((s_channel > s_thresh[0]) & (s_channel <= s_thresh[1]))
                 ] = 1
    return binary_output


# Set the width of the windows +/- margin
MARGIN = 75
# Set minimum number of pixels found to recenter window
MINPIX = 50

def visualize(original_img, warped_img, Minv, 
              fit_leftx, fit_rightx, fity, 
              nonzerox, nonzeroy, 
              left_lane_inds, right_lane_inds):

    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(original_img)
    # Color in left and right line pixels
    window_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    window_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([fit_leftx-MARGIN, fity]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_leftx+MARGIN, fity])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([fit_rightx-MARGIN, fity]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_rightx+MARGIN, fity])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    #result = cv2.addWeighted(warped_img, 1, window_img, 0.3, 0)

    img_size = window_img.shape[0:2]
    img_size = img_size[::-1] # Reverse order
    window_img = cv2.warpPerspective(window_img, Minv, img_size,
                                     flags=cv2.INTER_LINEAR)

    result = cv2.addWeighted(original_img, 1, window_img, 0.3, 0)

    if DEBUG:
        plt.imshow(result)
        plt.plot(fit_leftx, fity, color='yellow')
        plt.plot(fit_rightx, fity, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return result

def sliding_window_search(out_img, binimg, leftx_base, rightx_base):

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binimg.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binimg.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
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
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > MINPIX:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > MINPIX:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    return left_lane_inds, right_lane_inds

def optimized_search(binary_warped, left_fit, right_fit):

    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Return these as 1-item lists, for consistency with the return value of the sliding window function
    left_lane_inds  = [(((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - MARGIN)) 
                      & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + MARGIN))).nonzero()[0])]
    right_lane_inds = [(((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - MARGIN)) 
                      & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + MARGIN))).nonzero()[0])]

    return left_lane_inds, right_lane_inds

def pipeline(src_img):

    global OPTIMIZE, LEFT_FIT, RIGHT_FIT
    ret, mtx, dist, rvecs, tvecs = CAL_VARS

    #plt.imshow(src_img); plt.show()
    #print("Applying the distortion correction to the raw image.")
    original_image = cv2.undistort(src_img, mtx, dist, None, mtx)
    img_size = original_image.shape[0:2]
    img_size = img_size[::-1] # Reverse order
    
    #src = np.float32([[180,620],[1200,620],[840,450],[440,450]])
    src = np.float32([[(img_size[0]/2)-80,   img_size[1]/2+100],
                      [(img_size[0]/6)+10,   img_size[1]],
                      [(img_size[0]*5/6)-10, img_size[1]],
                      [(img_size[0]/2)+80,   img_size[1]/2+100]])
    #dst = np.float32([[0,720],[1280,720],[1280,0],[0,0]])
    dst = np.float32([[(img_size[0]/4), 0],
                      [(img_size[0]/4), img_size[1]],
                      [(img_size[0]*3/4), img_size[1]],
                      [(img_size[0]*3/4), 0]])

    # One especially smart way to do this would be to use 
    # four well-chosen corners that were automatically detected during the 
    # undistortion steps
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_image = cv2.warpPerspective(original_image, M, img_size,
                                       flags=cv2.INTER_LINEAR)
    if DEBUG: print('warped_image'); plt.imshow(warped_image); plt.show()

    #print("Using color transforms, gradients, etc.")
    #print("to create a thresholded binary image.")
    # Choose a Sobel kernel size
    ksize = 11 
    # Choose an odd number >= 3 to smooth gradient measurements
    gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray_warped, orient='x', 
                             sobel_kernel=ksize, 
                             thresh_min=50, thresh_max=200)
    if DEBUG: print('abs_sobel_thresh x'); plt.imshow(gradx); plt.show()
    #grady = np.zeros_like(gradx) \
    grady = abs_sobel_thresh(gray_warped, orient='y', 
                             sobel_kernel=ksize, 
                             thresh_min=50, thresh_max=200)
    if DEBUG: print('abs_sobel_thresh y'); plt.imshow(grady); plt.show()
    mag_binary = mag_thresh(gray_warped, sobel_kernel=ksize, 
                            mag_thresh=(50, 250))
    if DEBUG: print('mag_thresh'); plt.imshow(mag_binary); plt.show()
    #dir_binary = dir_threshold(gray_warped, sobel_kernel=ksize, 
    #                           thresh=(np.pi/2-0.1, np.pi/2+0.1))
    #if DEBUG: print('dir_binary'); plt.imshow(dir_binary); plt.show()
    hsv_binary_y = hsv_select(warped_image, 
                              h_thresh=(0,50), v_thresh=(100,255), s_thresh=(100,255))
    hsv_binary_w = hsv_select(warped_image, 
                              h_thresh=(20,255), v_thresh=(180,255), s_thresh=(0,80))
    if DEBUG: print('hls_select_y'); plt.imshow(hsv_binary_y); plt.show()
    if DEBUG: print('hls_select_w'); plt.imshow(hsv_binary_w); plt.show()
    combined = np.zeros_like(gradx) 
    combined[((gradx == 1) | (grady == 1)) 
             | ((mag_binary == 1))# & (dir_binary == 1)) 
             | ((hsv_binary_y == 1) | (hsv_binary_w == 1)) # Yellow and White
            ] = 1
    if DEBUG: print('Combined warped binary img'); plt.imshow(combined); plt.show()

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = combined.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    #print("Detecting lane pixels and fitting to find lane boundary.")
    histogram = np.sum(combined[combined.shape[0]/2:,:], axis=0)
    if DEBUG: plt.plot(histogram); plt.show()
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((combined, combined, combined))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    if not OPTIMIZE:
        left_lane_inds, right_lane_inds = sliding_window_search(out_img, combined, leftx_base, rightx_base)
        OPTIMIZE = True
    else:
        left_lane_inds, right_lane_inds = optimized_search(combined, LEFT_FIT, RIGHT_FIT)
                          
    if DEBUG: print("left_lane_inds, right_lane_inds = %s, %s" % (left_lane_inds, right_lane_inds))                             
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    LEFT_FIT = np.polyfit(lefty, leftx, 2)
    RIGHT_FIT = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    fity = np.linspace(0, combined.shape[0]-1, combined.shape[0] )
    fit_leftx = LEFT_FIT[0]*fity**2 + LEFT_FIT[1]*fity + LEFT_FIT[2]
    fit_rightx = RIGHT_FIT[0]*fity**2 + RIGHT_FIT[1]*fity + RIGHT_FIT[2]

    if DEBUG: print("Warp the detected lane boundaries back onto the original image.")
    result = visualize(original_image, warped_image, Minv,
                       fit_leftx, fit_rightx, fity, 
                       nonzerox, nonzeroy,
                       left_lane_inds, right_lane_inds)

    if DEBUG:
        print("Output visual display of the lane boundaries and numerical estimation ")
        print("of lane curvature and vehicle position.")
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
    if DEBUG:
        images = glob.glob('test_images/test*.jpg')
        print("Looking at images %s" % images)
        for idx, fname in enumerate(images):
            image = cv2.imread(fname)
            pipeline(image)
            cv2.imwrite('output_images/%s' % fname, image)
        # NOTE: save example images from each stage of your pipeline to the output_images folder 
        #       and provide a description of what each image is in your README for the project.
    else:

        # Run your algorithm on a video. In the case of the video, you must search for the lane lines in the first few frames, 
        # and, once you have a high-confidence detection, use that information to track the position and curvature of the lines 
        # from frame to frame. Save your output video and include it with the submission.
  
        if not TRY_CHALLENGE:
            output_file = 'project_video_out.mp4'
            clip = VideoFileClip('project_video.mp4')
            out_clip = clip.fl_image(pipeline)
            out_clip.write_videofile(output_file, audio=False)
        else:
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
