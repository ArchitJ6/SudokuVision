from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import cv2
import imutils

# Define the puzzle processing function
# This function will take an input image and return a 2-tuple of the puzzle in both RGB and grayscale
# The function will also take an optional debug flag that will display intermediate steps of the image processing pipeline
def find_puzzle(image, debug=False):
	# Convert the image to grayscale and blur it slightly
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7, 7), 3)
	
    # Apply adaptive thresholding and then invert the threshold map
	# thresh = cv2.adaptiveThreshold(blurred, 255,
	# 	cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	thresh = cv2.adaptiveThreshold(blurred, 255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
	thresh = cv2.bitwise_not(thresh)
	
	# Check to see if we are visualizing each step of the image
	# processing pipeline (in this case, thresholding)
	if debug:
		cv2.imshow("Puzzle Thresh", thresh)
		cv2.waitKey(0)
		
    # Find contours in the thresholded image and sort them by size in descending order
	# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	# 	cv2.CHAIN_APPROX_SIMPLE)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	
	# Initialize a contour that corresponds to the puzzle outline
	puzzleCnt = None
	
	# Loop over the contours
	for c in cnts:
		# Approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		
		# If our approximated contour has four points, then we can assume we have found the outline of the puzzle
		if len(approx) == 4:
			puzzleCnt = approx
			break
	
    # If the puzzle contour is empty then our script could not find the outline of the Sudoku puzzle
	if puzzleCnt is None:
		raise Exception(("Could not find Sudoku puzzle outline. "
			"Try debugging your thresholding and contour steps."))
	
	# Check to see if we are visualizing the outline of the detected Sudoku puzzle
	if debug:
		# Draw the contour of the puzzle on the image and then display it to our screen for visualization
		output = image.copy()
		cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
		cv2.imshow("Puzzle Outline", output)
		cv2.waitKey(0)
		
    # Apply a four point perspective transform to both the original image and grayscale image
    # to obtain a top-down bird's eye view of the puzzle
	puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
	warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
	
	# Check to see if we are visualizing the perspective transform
	if debug:
		# Show the output warped image (again, for debugging purposes)
		cv2.imshow("Puzzle Transform", puzzle)
		cv2.waitKey(0)
		
	# Return a 2-tuple of puzzle in both RGB and grayscale
	return (puzzle, warped)

# Define the extract_digit function
# This function will take an input cell and return the digit
# The function will also take an optional debug flag that will display intermediate steps of the image processing pipeline
def extract_digit(cell, debug=False):
	# Apply automatic thresholding to the cell and then clear any connected borders that touch the border of the cell
	thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	thresh = clear_border(thresh)
	
	# Check to see if we are visualizing the cell thresholding step
	if debug:
		cv2.imshow("Cell Thresh", thresh)
		cv2.waitKey(0)
	
    # Find contours in the thresholded cell
    # and sort them by size in descending order
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	
    # If no contours were found then we are examining an empty cell
	if len(cnts) == 0:
		return None
	
    # Otherwise, find the largest contour in the cell and create a mask for the contour
	c = max(cnts, key=cv2.contourArea)
	mask = np.zeros(thresh.shape, dtype="uint8")
	cv2.drawContours(mask, [c], -1, 255, -1)
	
    # Compute the percentage of masked pixels in the contour region
	(h, w) = thresh.shape
	percentFilled = cv2.countNonZero(mask) / float(w * h)
	
	# If less than 3% of the mask is filled then we are looking at noise and can safely ignore the contour
    # Otherwise, apply the mask to the thresholded cell
	if percentFilled < 0.03:
		return None
	
	# Apply the mask to the thresholded cell
    # to get the digit
	digit = cv2.bitwise_and(thresh, thresh, mask=mask)
	
    # Check to see if we are visualizing the masking step
	if debug:
		cv2.imshow("Digit", digit)
		cv2.waitKey(0)
		
	# Return the digit to the calling function
	return digit