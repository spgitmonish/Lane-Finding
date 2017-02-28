import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def area_in_focus(img, vertices):
    """Shows the area in focus for processing"""
    # Copy the contents of image(Note imageInFocus=image will point to the same array not the contents)
    imageInFocus = cv2.polylines(img, [vertices], True, (0, 0, 255), 3, 4)
    return

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        x1 = int(line[0])
        y1 = int(line[1])
        x2 = int(line[2])
        y2 = int(line[3])
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def process_line_segments(lines, imshape):
    """
    Function to process the hough transform line segments to get the
    lines to be drawn on the final image to represent the lines on the road
    """
    # Variables to store the slopes of the right and left line segments
    rightSlope = np.array([ ])
    leftSlope = np.array([ ])

    # Variables to store points which is used for line fitting the
    # right and left lines
    rLineSegPoints = np.empty(shape=[0, 2])
    lLineSegPoints = np.empty(shape=[0, 2])

    # Points which represent the max and minimum points on the
    # right line which is to be drawn
    rxMax = imshape[1]
    ryMax = imshape[0]
    rxMin = ryMin = 0

    # Points which represent the max and minimum points on the
    # left line which is to be drawn
    lxMax = imshape[1]
    lyMax = imshape[0]
    lxMin = lyMin = 0

    globalYMin = imshape[1]

    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/float((x2-x1))
            # Right sided line segments
            if slope >= 0.5 and (np.isfinite(slope) == True):
                # Update the global y minimum
                if y1 <= globalYMin:
                    globalYMin = y1
                elif y2 <= globalYMin:
                    globalYMin = y2

                # Append the points to rlineSegments to fit the right line
                rLineSegPoints = np.append([[x1, y1]], rLineSegPoints, axis=0)
                rLineSegPoints = np.append([[x2, y2]], rLineSegPoints, axis=0)
                rightSlope = np.append(slope, rightSlope)

            # Left sided line segments
            elif slope <= -0.5 and (np.isfinite(slope) == True):
                # Update the global y minimum
                if y1 <= globalYMin:
                    globalYMin = y1
                elif y2 <= globalYMin:
                    globalYMin = y2

                # Append the points to rlineSegments to fit the left line
                lLineSegPoints = np.append([[x1, y1]], lLineSegPoints, axis=0)
                lLineSegPoints = np.append([[x2, y2]], lLineSegPoints, axis=0)
                leftSlope = np.append(slope, leftSlope)

    # Get the average of the slopes of the left and right side line segments
    rightSlope = np.mean(rightSlope)
    leftSlope = np.mean(leftSlope)

    # NOTE: vrx, vry represents a vector which represents the direction of the line
    # Use the best fit line points and determine intercept b
    [vrx, vry, rx, ry] = cv2.fitLine(rLineSegPoints, cv2.DIST_FAIR, 0, 0.01, 0.01)
    rb = rightSlope * rx - ry

    # Use the best fit line points and determine intercept b
    [vlx, vly, lx, ly] = cv2.fitLine(lLineSegPoints, cv2.DIST_FAIR, 0, 0.01, 0.01)
    lb = leftSlope * lx - ly

    # The minium should be the same on both sides to have even lanes
    ryMin = lyMin = globalYMin

    # Right sided minimum and maximum
    rxMin = (ryMin + rb)/rightSlope
    rxMax = (ryMax + rb)/rightSlope

    # Left sided minimum and maximum
    lxMin = (lyMin + lb)/leftSlope
    lxMax = (lyMax + lb)/leftSlope

    # Return lines to be drawn
    linesToBeDrawn = np.empty(shape=[2,4], dtype=np.float)
    linesToBeDrawn = np.append([[rxMin, ryMin, rxMax, ryMax]], linesToBeDrawn, axis=0)
    linesToBeDrawn = np.append([[lxMin, lyMin, lxMax, lyMax]], linesToBeDrawn, axis=0)

    return linesToBeDrawn

def color_filter_image(image):
    """ Function which performs additional filtering to improve detection of
        of yellow and white lines/dashes on the road before applying grayscale
    """
    # Make copies of the image for filtering
    imageCopy = np.copy(image)
    imageCopy1 = np.copy(image)
    imageCopy2 = np.copy(image)

    # Select RGB ranges for yellow
    lower_yellow = np.array([200, 150, 0])
    upper_yellow = np.array([255, 255, 100])

    # Select RGB ranges for white
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])

    # Filter only by yellow
    maskY = cv2.inRange(imageCopy1, lower_yellow, upper_yellow)
    imageCopy1 = cv2.bitwise_and(imageCopy1, imageCopy1, mask=maskY)

    # Filter only by white
    maskW = cv2.inRange(imageCopy2, lower_white, upper_white)
    imageCopy2 = cv2.bitwise_and(imageCopy2, imageCopy2, mask=maskW)

    # Logically OR the masks
    mask = maskY | maskW

    # Get the new filtered image
    imageCopy = cv2.bitwise_and(imageCopy, imageCopy, mask=mask)
    return imageCopy

def process_image(image, showImages=0, frameCount=0, saveImages=0):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    # Read in JPG for images
    """image1 = (mpimg.imread('test_images/solidWhiteCurve.jpg')).astype('uint8')
    image2 = (mpimg.imread('test_images/solidWhiteRight.jpg')).astype('uint8')
    image3 = (mpimg.imread('test_images/solidYellowCurve.jpg')).astype('uint8')
    image4 = (mpimg.imread('test_images/solidYellowCurve2.jpg')).astype('uint8')
    image5 = (mpimg.imread('test_images/solidYellowLeft.jpg')).astype('uint8')
    image6 = (mpimg.imread('test_images/whiteCarLaneSwitch.jpg')).astype('uint8')
    image = np.copy(image6)"""
    if showImages == 1:
        plt.title("Original")
        plt.imshow(image)
        plt.show()

    # Apply color filter to enhance yellow and white detection
    imageCopy = color_filter_image(image)
    if showImages == 1:
        plt.title("Filtered image")
        plt.imshow(image)
        plt.show()

    # Gray Scale the image selected
    gray = grayscale(imageCopy)
    if showImages == 1:
        plt.title("Gray Scale")
        plt.imshow(gray, cmap='gray')
        plt.show()

    # Apply gaussian smoothing with the kernel size set to 5
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    if showImages == 1:
        plt.title("Blur Image")
        plt.imshow(blur_gray, cmap='gray')
        plt.show()

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Define vertices for a polygon for the area in focus
    # This time we are defining a four sided polygon to mask
    imshape = image.shape

    vy1 = np.int32(imshape[0] * 0.585)
    vx1 = np.int32(imshape[1] / 2)
    vy2 = vy1
    vx2 = vx1 + 10

    # Narrow down the polygon to as close to a triangle as possible but make it a "4 sided polygon"
    vertices = np.array([[(0,imshape[0]),(vx1, vy1), (vx2, vy2), (imshape[1],imshape[0])]], dtype=np.int32)
    imageInFocus = np.copy(imageCopy)
    area_in_focus(imageInFocus, vertices)

    # Masked Area of interest for applying the hough transform
    masked_edges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Distance resolution in pixels of the Hough grid
    rho = 2
    # Angular resolution in radians of the Hough grid
    theta = np.pi/180
    # Minimum number of votes (intersections in Hough grid cell)
    threshold = 15
    # Minimum number of pixels making up a line
    min_line_length = 20
    # Maximum gap in pixels between connectable line segments
    max_line_gap = 20

    # Run Hough on edge detected image
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    if saveImages == 0:
        # Using the line segments, process to get a single line to
        # be drawn on the final image
        final_lines = process_line_segments(lines, imshape)

        # Make a blank the same size as our image to draw on
        line_image = np.copy(image)*0
        line_image = draw_lines(line_image, final_lines)

        # Draw the lines on the original image
        line_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
        if showImages == 1:
            plt.title("Lines")
            plt.imshow(line_edges)
            plt.show()
    else:
        # Make a blank the same size as our image to draw on
        line_image = np.copy(image)*0
        process_line_segments(lines, imshape)

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 6)

        # Draw the lines on the original image
        line_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
        if showImages == 1:
            plt.title("Lines")
            plt.imshow(line_edges)
            plt.show()
        fileName = "HoughOutput%d.png" % frameCount
        mpimg.imsave(fileName, line_edges)

    return line_edges

def process_all_videos():
    # Get all the files required and process each video
    white_output = 'whiteProcessed.mp4'
    clip = VideoFileClip("solidWhiteRight.mp4")
    #NOTE: this function expects color images!!
    white_clip = clip.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)

    yellow_output = 'yellowProcessed.mp4'
    clip = VideoFileClip("solidYellowLeft.mp4")
    yellow_clip = clip.fl_image(process_image)
    yellow_clip.write_videofile(yellow_output, audio=False)

    challenge_output = 'challengeProcessed.mp4'
    clip = VideoFileClip("challenge.mp4")
    challenge_clip = clip.fl_image(process_image)
    challenge_clip.write_videofile(challenge_output, audio=False)

process_all_videos()
