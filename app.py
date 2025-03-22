import streamlit as st
import cv2
import tempfile
import numpy as np
from solve import solve  # Importing solve function

st.title("🧩 Sudoku Solver")

uploaded_file = st.file_uploader("Upload an image of a Sudoku puzzle", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name

    image = cv2.imread(temp_path)
    if image is None:
        st.error("Invalid image. Please upload a clear Sudoku puzzle image.")
    else:
        st.image(image, caption="Uploaded Sudoku Image", use_container_width =True)

        if st.button("Solve Sudoku"):
            with st.spinner("Processing..."):
                result, error, board = solve(temp_path)

            if error:
                if board:
                    s = ["+-------+-------+-------+"]  # Initialize list to store rows of the board
                    for i in range(9):
                        row = ['|']  # Initialize list to store columns of the board
                        for j in range(9):
                            if board[i][j] == 0:
                                row.append(" ")
                            else:
                                row.append(str(board[i][j]) if board[i][j] is not None else " ")

                            if j == 2 or j == 5:
                                row.append("|")  # Align columns with separators
                        row.append("|") # Add right border
                        s.append(" ".join(row))  # Convert row list to string

                        if i == 2 or i == 5:
                            s.append("+-------+-------+-------+")  # Add row separator
                    s.append("+-------+-------+-------+")  # Add bottom border
                    sudoku_board = "\n".join(s)  # Join all rows to form the board

                st.error(error)
                st.code(sudoku_board, language="plaintext")  # Use st.code() to maintain formatting
            else:
                st.image(result, caption="Solved Sudoku", use_container_width =True)