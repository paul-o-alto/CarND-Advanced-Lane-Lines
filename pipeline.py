import glob
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

DEBUG = True

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (8,6), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            #cv2.imshow('img', img)
            #cv2.waitKey(1) #500)

    cv2.destroyAllWindows()
    #print("objpoints = %s" % objpoints)
    #print("imgpoints = %s" % imgpoints)
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
 
# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def project_onto(unwarped, warped, Minv, 
                 left_fitx, right_fitx, yvals):

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))*255

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left  = np.array([ np.transpose(np.vstack([left_fitx , yvals]))])
    pts_right = np.array([np.flipud(
                           np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using 
    # inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(
                  color_warp, Minv, image.shape[0:2]) 
    # Combine the result with the original image
    result = cv2.addWeighted(unwarped, 1, newwarp, 0.3, 0)
    plt.imshow(result)

    return result

def sliding_window_search(binary_warped, leftx_base, rightx_base):

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binimg.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binimg.shape[0] - (window+1)*window_size
        win_y_high = binimg.shape[0] - window*window_size
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
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
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

def optimized_search(binary_warped, left_fit, right_fit):

    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds  = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) 
                     & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) 
                     & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    return left_lane_inds, right_lane_inds

def pipeline(cal_vars, src_img):

    ret, mtx, dist, rvecs, tvecs = cal_vars

    #plt.imshow(src_img); plt.show()
    print("Applying the distortion correction to the raw image.")
    image = cv2.undistort(src_img, mtx, dist, None, mtx)
    plt.imshow(image); plt.show() 
    #corners = cv2.undistortPoints(image, mtx, dist)    



    print("Applying a perspective transform to rectify binary image")
    print(" (make it a birds-eye view)")
    # NOTE: Need to find src and dst points, perhaps automatically? 
    #       can we use some static ones and be correct?
    #src = corners[0:4] 720 1280 y, x
    src = np.float32([[360,640],[720,100],[720,1180],[360,740]])
    # NOTE: you could pick any four of the detected corners 
    # as long as those four corners define a rectangle
    # One especially smart way to do this would be to use 
    # four well-chosen
    # corners that were automatically detected during the 
    # undistortion steps
    #We recommend using the automatic detection of corners in your code
    #dst = corners[0:4] 
    dst = np.float32([[0,0],[720,0],[720,1280],[0,1280]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = image.shape[0:2]
    warped_image = cv2.warpPerspective(image, M, img_size,
                                       flags=cv2.INTER_LINEAR)


    print("Using color transforms, gradients, etc.")
    print("to create a thresholded binary image.")
    # Choose a Sobel kernel size
    ksize = 11 
    # Choose an odd number >= 3 to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(warped_image, orient='y', 
                             sobel_kernel=ksize, 
                             thresh_min=50, thresh_max=200)
    #plt.imshow(gradx); plt.show()
    grady = abs_sobel_thresh(warped_image, orient='x', 
                             sobel_kernel=ksize, 
                             thresh_min=50, thresh_max=200)
    #plt.imshow(grady); plt.show()
    mag_binary = mag_thresh(warped_image, sobel_kernel=ksize, 
                            mag_thresh=(50, 200))
    #plt.imshow(mag_binary); plt.show()
    dir_binary = dir_threshold(warped_image, sobel_kernel=ksize, 
                               thresh=(1, np.pi/2))
    hls_binary = hls_select(warped_image, thresh=(100, 255))
    #plt.imshow(hls_binary); plt.show()
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | 
             ((mag_binary == 1) & (dir_binary == 1)) |  
             (hls_binary == 1)] = 1
    plt.imshow(combined)
    plt.show()


    print("Detecting lane pixels and fitting to find lane boundary.")
    histogram = np.sum(combined[combined.shape[0]/2:,:], axis=0)
    if DEBUG: plt.plot(histogram); plt.show()

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    #search_y = np.copy(yvals)
    #leftx  = np.argmax(search_y)
    #rightx = np.argmax(np.flipud(search_y))
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #if rightx == leftx:
    #    search_y[rightx] = 0
    #    rightx = np.argmax(np.flipud(search_y)) 
    if not optimize:
        left_lane_inds, right_lane_inds = sliding_window_search(combined, leftx_base, rightx_base)
    else:
        left_lane_inds, right_lane_inds = optimized_search(binary_warped, left_fit, right_fit)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if DEBUG:
        # Generate x and y values for plotting
        fity = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
        fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(fit_leftx, fity, color='yellow')
        plt.plot(fit_rightx, fity, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)


    print("Warp the detected lane boundaries back onto the original image.")
    result = project_onto(image, warped_image, Minv, 
                          left_fitx, right_fitx, yvals)

    print("Output visual display of the lane boundaries and numerical estimation ")
    print("of lane curvature and vehicle position.")
    if DEBUG: plt.imshow(result); plt.show()
    return result #TODO: Finish this section

def process_image():
    # TODO: Implement a generator for video clips?
    # Would allow for initial sliding window and then optimized version
    pass

def main():

    # So we don't have to calibrate every time
    try:
        with open('./cal_vars.p', 'rb') as _input:
            cal_vars = pickle.load(_input)
        print('Successfully loaded prior calibration variables')
    except Exception as e:
        print('Got exception %s when trying to load calibration variables' % e)
        print('No saved calibration variables, calibrating now (saving for future use)')
        cal_vars = calibration()
        with open('./cal_vars.p', 'wb') as output:
            pickle.dump(cal_vars, output)    
        
 
    # TODO: For a series of test images (in the test_images folder in the repository): 
    images = glob.glob('test_images/test*.jpg')
    print("Looking at images %s" % images)
    for idx, fname in enumerate(images):
        image = cv2.imread(fname)
        pipeline(cal_vars, image, optimize=False)
    # NOTE: save example images from each stage of your pipeline to the output_images folder 
    #       and provide a description of what each image is in your README for the project.


    # Run your algorithm on a video. In the case of the video, you must search for the lane lines in the first few frames, 
    # and, once you have a high-confidence detection, use that information to track the position and curvature of the lines 
    # from frame to frame. Save your output video and include it with the submission.
    #white_output = 'white.mp4'
    #clip1 = VideoFileClip("solidWhiteRight.mp4")
    #white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!


if __name__ == "__main__":
    main()
