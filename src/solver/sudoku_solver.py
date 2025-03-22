from src.utils.validation import is_valid_sudoku, is_valid_solution, board_to_sudoku

def solve_sudoku(board):
    # Construct a Sudoku puzzle from the board
    print("[INFO] OCR'd Sudoku board:")
    puzzle = board_to_sudoku(board)
    puzzle.show()

    # Check if the Sudoku board is valid
    if not is_valid_sudoku(board):
        print("[INFO] OCR'd Sudoku board is not valid!")
        return None

    # Solve the Sudoku puzzle
    print("[INFO] solving Sudoku puzzle...")
    solution = puzzle.solve()
    solution.show_full()

    # Check if the solution is valid
    if not is_valid_solution(solution.board):
        print("[INFO] solution is invalid")
        return None

    # Return the solution as a numpy array
    return solution