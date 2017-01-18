import glob
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

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

def project_onto(img):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

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
                  color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)

def pipeline(cal_vars, src_img):

    # TODO: LOOK BACK AT PROJECT 1 for more useful code
    ret, mtx, dist, rvecs, tvecs = cal_vars

    # 1.) Apply the distortion correction to the raw image.
    image = cv2.undistort(src_img, mtx, dist, None, mtx)
    #plt.imshow(image); plt.show()
    #corners = cv2.undistortPoints(image, mtx, dist)    

    # 2.) Use color transforms, gradients, etc., to create a thresholded binary image.
    # Choose a Sobel kernel size
    ksize = 9 # Choose a larger odd number to smooth gradient measurements
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh_min=1, thresh_max=255)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh_min=1, thresh_max=255)
    mag_binary = mag_thresh(   image, sobel_kernel=ksize, mag_thresh=(1, 255))
    #plt.imshow(gradx); plt.show()
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.01, np.pi/2))
    hls_binary = hls_select(image, thresh=(1, 255))
    plt.imshow(hls_binary); plt.show()
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1
    img_size = combined.shape
    plt.imshow(combined)
    plt.show()

    # 3.) Apply a perspective transform to rectify binary image ("birds-eye view").
    # NOTE: Need to find src and dst points, perhaps automatically? can we use some static ones and be correct?
    #src = corners[0:4] 720 1280 y, x
    src = np.float32([[360,640],[720,100],[720,1180],[360,740]])
    # NOTE: you could pick any four of the detected corners 
    # as long as those four corners define a rectangle
    #One especially smart way to do this would be to use four well-chosen
    # corners that were automatically detected during the undistortion steps
    #We recommend using the automatic detection of corners in your code
    #dst = corners[0:4] 
    dst = np.float32([[0,0],[720,0],[720,1280],[0,1280]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)        

    # 4.) TODO: Detect lane pixels and fit to find lane boundary. 
    histogram = np.sum(combined[combined.shape[0]/2:,:], axis=0)
    plt.plot(histogram)
    plt.show()

    # 5.) Determine curvature of the lane and vehicle position with respect to center.
    # Fit a second order polynomial to each fake lane line
    left_fit = np.polyfit(yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
    right_fit = np.polyfit(yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]
    # Plot up the data # FOR DEBUG
    plt.plot(leftx, yvals, 'o', color='red')
    plt.plot(rightx, yvals, 'o', color='blue')
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, yvals, color='green', linewidth=3)
    plt.plot(right_fitx, yvals, color='green', linewidth=3)
    plt.gca().invert_yaxis() # to visualize as we do the images
    y_eval = np.max(yvals)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) \
                             /np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) \
                                /np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)


    # 6.) Warp the detected lane boundaries back onto the original image.
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)

    # 7.) Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    return result #TODO: Finish this section

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
        pipeline(cal_vars, image)
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
