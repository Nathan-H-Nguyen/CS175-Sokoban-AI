import sys
from collections import deque
from typing import Set, Tuple

class Board:
    """
    Represents the Sokoban game board.

    Parses an input file and initializes the board with walls, boxes, storage locations,
    and the starting agent position. Board is a 2D-Matrix.
    """

    ########### INITIALIZATION ##########
    def __init__(self, file: str) -> None:
        """
        Initializes the Board object.

        Args:
            filename (str): Path to the Sokoban input file.
        
        Returns:
            None
        """
        self.file = file # Input file

        self.rows = 0 # Number of rows in the board
        self.cols = 0 # Number of columns in the board
        self.board = [] # 2D-Matrix for board itself

        self.num_walls = 0 # Total number of walls
        self.walls = set() # Set containing tuples() of the (x,y) coordinates

        self.num_boxes = 0 #Total number of boxes
        self.boxes = set() # Set containing tuples() of the (x,y) coordinates

        self.num_storages = 0 # Total number of storage locations
        self.storages = set() # Set containing tuples() of the (x,y) coordinates

        self.starting_pos = tuple() # Starting (x,y) location
        self.player_pos = tuple() # Current player (x,y) location

        # Initialize data members and board
        self._initialize_data_members()
        self._initialize_game_board()


    @staticmethod
    def _get_target_coordinate(coord: tuple, direction: str):
        x, y  = coord
        if direction == 'L':
            return x, y - 1
        elif direction == 'R':
            return x, y + 1
        elif direction == 'U':
            return x - 1, y
        elif direction == 'D':
            return x + 1, y

        raise ValueError(
            f"Invalid direction: {direction}. Must be one of 'L', 'R', 'U', or 'D'.")


    def _initialize_data_members(self) -> None:
        """
        Parses input file and initializes all data members other than board itself.

        Returns:
            None
        """

        try:
            with open(self.file, 'r') as f:
                content = f.read().splitlines()

            def decodeLine(line):
                data = list(map(int, line.split()))
                return data[0], set((data[i], data[i + 1]) for i in range(1, len(data), 2))

            self.rows, self.cols = map(int, content[0].split())
            self.num_walls, self.walls = decodeLine(content[1])
            self.num_boxes, self.boxes = decodeLine(content[2])
            self.num_storages, self.storages = decodeLine(content[3])
            x, y = map(int, content[4].split())
            self.starting_pos = (x - 1, y - 1)
            self.player_pos = (x - 1, y - 1)

        except Exception as e:
            print(f"Error: {e}")

    def _initialize_game_board(self) -> None:
        """
        Initializes the game board as a 2D-Matrix.

        Returns:
            None
        """

        # Initialize empty board
        self.board = [[' ']*self.cols for row in range(self.rows)]

        # Place Walls
        for x, y in self.walls:
            self.board[x][y] = '#'

        # Place Boxes
        for x, y in self.boxes:
            self.board[x][y] = '$'
        
        # Place Storage Locations
        for x, y in self.storages:
            self.board[x][y] = '.'
        
        # Place Player
        self.board[self.starting_pos[0]][self.starting_pos[1]] = '@'

    ########### PUBLIC METHODS ##########
    def move(self, direction: str) -> bool:
        """
        Attempts to move the player in the specified direction.
        
        Returns:
            bool: True if player moved, False otherwise
        """
        x, y = self.player_pos # Get current player coordinates

        # Get coordinate we are trying to move to
        move_coordinate = self._get_target_coordinate((x,y), direction)
        
        # If square is a wall, then we can't move and return false
        if move_coordinate in self.walls:
            return False
        
        # If square is a box attempt to push and move in that spot
        if move_coordinate in self.boxes:
            if not self.push_box(move_coordinate, direction): # Attempt to Push box
                return False
        
        # Remove player from current space
        if self.player_pos in self.storages:
            self.board[x][y] = '.'
        else:
            self.board[x][y] = ' '

        # Move player into new space
        self.board[move_coordinate[0]][move_coordinate[1]] = '@'
        self.player_pos = move_coordinate

        return True

    def is_win(self) -> bool:
        """
        Checks if the current game state has won, i.e. all boxes on storage locations

        Returns:
            bool: Returns True upon win, False otherwise
        """

        # Check if all boxes are in storage locations, if no return False
        for box in self.boxes:
            if box not in self.storages:
                return False
            
        return True

    def print(self) -> None:
        """
        Prints out the current Board to the console

        Returns:
            None
        """

        for row in self.board:
            line = ''
            for col in row:
                line += col
            print(line)
    
    ########### PRIVATE HELPERS ##########
    def push_box(self, coordinate: Tuple[int,int], direction: str) -> bool:
        """
        Attempts to push the specified box in the given direction.

        Args:
            coordinate (Tuple[int,int]): Tuple representing the coordinates of the box to be pushed (x,y).
            direction (str): Str representing the direction to push the box: 'L', 'R', 'U', 'D'.
        
        Returns:
            bool: True if box was pushed, False otherwise

        Raises:
            ValueError: If an invalid direction is passed as an argument.
        """
        x, y = coordinate

        # Get coordinate we are trying to push box to
        pushed_coordinate = self._get_target_coordinate(coordinate, direction)

        # If a Wall or Box is blocking that direction, then we can't push
        if pushed_coordinate in self.walls or pushed_coordinate in self.boxes:
            return False
        
        # Push box
        if coordinate in self.storages:
            self.board[x][y] = '.'
        else:
            self.board[x][y] = ' '
        self.board[pushed_coordinate[0]][pushed_coordinate[1]] = '$'
        self.boxes.remove(coordinate) # Remove old coordinate
        self.boxes.add(pushed_coordinate) # Add new coordinate

        return True

    def box_corner_trap(self) -> bool:
        """
        Checks if any box is trapped in a corner and not on a storage location.

        Returns:
            bool: True if a box is trapped in a corner, else False
        """
        for box in self.boxes:
            # If box in storage location skip, no need to check if trapped
            if box in self.storages:
                continue
            
            x, y =  box
            left = (x, y-1)
            right = (x, y+1)
            up = (x-1, y)
            down = (x+1, y)

            if up in self.walls and right in self.walls: # Top right corner
                return True
            elif right in self.walls and down in self.walls: # Bottom right corner
                return True
            elif down in self.walls and left in self.walls: # Bottom left corner
                return True
            elif left in self.walls and up in self.walls: # Top left corner
                return True
            
        return False
    
if __name__ == '__main__':
    board = Board(sys.argv[1])
    valid = {'L', 'R', 'U', 'D'}
    lose_flag = 0

    board.print()
    while not board.is_win():

        direction = input("Enter a direction to move (L, R, U, D): ").strip().upper()
        if direction not in valid:
            print(f"Invalid direction '{direction}'.")
            continue
        board.move(direction)
        board.print()

        if board.box_corner_trap():
            lose_flag = 1
            break
    
    if not lose_flag:
        print("Game Won!")
    else:
        print("Game Lost due to deadlock.")