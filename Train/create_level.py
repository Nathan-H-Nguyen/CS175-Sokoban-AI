import os
import time
# Easy => <7x7 World with 3 boxes or less
# Medium => 10x10 World with 4 boxes
# Hard => >10x10 World with 5 boxes or more

def input_to_sokoban(filename: str) -> None:
    """
    Takes a file and turns it into a Sokoban file valid for input

    Args:
        filename (str): Filename/path
    
    Returns:
        None
    """
    rows = 0
    cols = 0

    num_walls = 0
    walls = []

    num_boxes = 0
    boxes = []

    num_storages = 0
    storages = []

    player = None

    board = []

    # Turn representation into 2D matrix
    with open(f'Train/Input/{filename}', 'r') as f:
        lines = f.readlines()
        for line in lines:
            board.append(list(line.rstrip('\n')))

    # Iterate through board in row major
    for row in range(len(board)):
        row_char = []
        row_length = len(board[row])

        # Update Rows and Columns
        rows += 1
        if row_length > cols:
            cols = len(board[row])

        for col in range(row_length):
            char = board[row][col]
            row_char.append(char)

            # Keep track of number of walls and location
            if char == '#':
                num_walls += 1
                walls.append((row+1, col+1))
            
            # Keep track of number of boxes and location
            if char == '$':
                num_boxes += 1
                boxes.append((row+1, col+1))
            
            # Keep track of number of storages and location
            if char == '.':
                num_storages += 1
                storages.append((row+1, col+1))
            
            if char == '@':
                player = (row+1, col+1)
        
        print(''.join(row_char))
    
    # Create valid input file
    with open(f'Train/Sokoban/sokoban-{filename}', 'w') as f:
        # First line: rows cols
        f.write(f'{rows} {cols}\n')

        # Second line: num_walls locations
        f.write(f'{num_walls} ')
        for x,y in walls:
            f.write(f'{x} {y} ')
        f.write('\n')

        # Third line: num_boxes locations
        f.write(f'{num_boxes} ')
        for x,y in boxes:
            f.write(f'{x} {y} ')
        f.write('\n')

        # Fourth line: num_storages locations
        f.write(f'{num_storages} ')
        for x,y in storages:
            f.write(f'{x} {y} ')
        f.write('\n')
    
        # Fifth line: player
        f.write(f'{player[0]} {player[1]}')
    
if __name__ == '__main__':
    input_path = os.path.join(os.getcwd(), 'Train/Input')
    
    for file in os.listdir(input_path):
        print(f'FILE: {file}')
        input_to_sokoban(file)
        time.sleep(1)