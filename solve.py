from src.ocr.puzzle_processing import find_puzzle, extract_digit
from keras import preprocessing
from keras import models
from src.solver.sudoku_solver import solve_sudoku
import cv2
import numpy as np
import argparse
import imutils
from src.utils.validation import board_to_sudoku

# Load the digit classifier
print("[INFO] loading digit classifier...")
# model = models.load_model(args["model"])
model = models.load_model("checkpoints/digit_classifier.keras")

def solve(img, debug=0, type="app"):
    print(img)
    # Check if the input image path exists
    if not cv2.os.path.exists(img):
        if type == "script":
            print(f"[ERROR] The input image path {img} does not exist")
            return
        else:
            return None, "Error in uploading the image. Please try again.", None
    
    # Load the input image from disk, resize it, and convert it to grayscale
    print("[INFO] processing image...")
    image = cv2.imread(img)
    image = imutils.resize(image, width=600)

    # Find the puzzle in the image and then extract the puzzle cells
    (puzzleImage, warped) = find_puzzle(image, debug=debug > 0)

    # Initialize our 9x9 Sudoku board
    board = np.zeros((9, 9), dtype="int")

    # A Sudoku puzzle consists of 9x9 grid, but we need to extract each cell from the grid
    # to classify the digits. We will divide the grid into 9x9 cells and then extract each cell
    # from the grid
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9

    # Initialize a list to store the (x, y)-coordinates of each cell location
    cellLocs = []

    # Loop over the grid locations
    for y in range(0, 9):
        # Initialize the current list of cell locations
        row = []
        for x in range(0, 9):
            # Compute the starting and ending (x, y)-coordinates of the current cell
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY

            # Add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))
            
            # Crop the cell from the warped transform image and then extract the digit from the cell
            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell, debug=debug > 0)
            
            # Verify that the digit is not empty
            if digit is not None:
                # Resize the cell to 28x28 pixels and then prepare the cell for classification
                # by converting it from an image to a floating point data type and scaling the pixel
                region_of_interest = cv2.resize(digit, (28, 28))
                region_of_interest = region_of_interest.astype("float") / 255.0
                region_of_interest = preprocessing.image.img_to_array(region_of_interest)
                region_of_interest = np.expand_dims(region_of_interest, axis=0)
                
                # classify the digit and update the Sudoku board with the prediction
                pred = model.predict(region_of_interest).argmax(axis=1)[0]
                board[y, x] = pred
        
        # Add the row to our cell locations
        cellLocs.append(row)
        
    solution = solve_sudoku(board)

    if solution is None:
        if type == "script":
            return
        else:
            sudoku_representation = board_to_sudoku(board).board
            return None, "Sudoku could not be solved. The Sudoku puzzle may be invalid. Sudoku board:", sudoku_representation

    # Loop over the cell locations and board
    for (cellRow, boardRow) in zip(cellLocs, solution.board):
        # Loop over individual cell in the row
        for (box, digit) in zip(cellRow, boardRow):
            # Unpack the cell coordinates
            startX, startY, endX, endY = box
            
            # Compute the coordinates of where the digit will be drawn on the output puzzle image
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.2)
            textX += startX
            textY += endY
            
            # Draw the result digit on the Sudoku puzzle image
            cv2.putText(puzzleImage, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # If the puzzle type is "script" then we will display the solution to the puzzle
    if type == "script":
        # Show the output image
        cv2.imshow("Sudoku Result", puzzleImage)
        cv2.waitKey(0)
    else:
        return puzzleImage, None, None

if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="checkpoints/digit_classifier.keras", help="path to trained digit classifier model")
    ap.add_argument("-i", "--image", required=True, help="path to input Sudoku puzzle image")
    ap.add_argument("-d", "--debug", type=int, default=-1, help="whether or not we are visualizing each step of the pipeline")
    args = vars(ap.parse_args())

    solve(args["image"], args["debug"], type="script")