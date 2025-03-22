from sudoku import Sudoku

def is_valid_solution(board):
    # Check rows
    for row in board:
        filtered_row = [num for num in row if num != 0 and num is not None]
        if len(filtered_row)==0:
            return False
        if len(set(filtered_row)) != len(filtered_row):  # Check for duplicates
            return False

    # Check columns
    for col in range(9):
        column = [board[row][col] for row in range(9)]
        filtered_column = [num for num in column if num != 0 and num is not None]
        if len(filtered_column)==0:
            return False
        if len(set(filtered_column)) != len(filtered_column):
            return False

    # Check 3x3 subgrids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid = []
            for x in range(i, i + 3):
                for y in range(j, j + 3):
                    subgrid.append(board[x][y])
            filtered_subgrid = [num for num in subgrid if num != 0 and num is not None]
            if len(set(filtered_subgrid)) != len(filtered_subgrid):
                return False

    return True

def is_valid_sudoku(board):
    # Check rows for duplicates, ignoring zeros
    for row in board:
        # Remove zeros before checking for duplicates
        row_no_zeros = [x for x in row if x != 0]
        if len(set(row_no_zeros)) != len(row_no_zeros):
            return False
    
    # Check columns for duplicates, ignoring zeros
    for col in range(9):
        column = [board[row][col] for row in range(9)]
        column_no_zeros = [x for x in column if x != 0]
        if len(set(column_no_zeros)) != len(column_no_zeros):
            return False
    
    # Check 3x3 subgrids for duplicates, ignoring zeros
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid = []
            for x in range(i, i + 3):
                for y in range(j, j + 3):
                    if board[x][y] != 0:
                        subgrid.append(board[x][y])
            if len(set(subgrid)) != len(subgrid):
                return False
    
    return True

def board_to_sudoku(board):
    # Convert the board to a Sudoku object
    return Sudoku(3, 3, board=board.tolist())