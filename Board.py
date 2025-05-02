import sys
from collections import deque
from typing import Set, Tuple

class Board:
    """
    Represents the Sokoban game board.

    Parses an input file and initializes the board with walls, boxes, storage locations,
    and the starting agent position. Board is a 2D-Matrix
    """

    def __init__(self, file: str) -> None:
        """
        Initializes the Board object

        Args:
            filename (str): Path to the Sokoban input file.
        
        Returns:
            None
        """
        self.file = file
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

        self._initialize_data_members()
        self._initialize_game_board()

    def _initialize_data_members(self) -> None:
        """
        Parses input file and initializes all data members other than board itself

        Returns:
            None
        """

        try:
            with open(self.file, 'r') as f:
                content = f.readlines()

                # Initialize rows and cols
                self.rows, self.cols = map(int, content[0].strip().split())

                # When getting x coordinates, get odd indexed elements
                # When getting y coordinates, get even indexed elements (Except for 0 index as that is the total number of elements)
                # Subtract x and y by 1 to account for 0 indexing

                # Initialize num_walls and walls
                wall_line = list(map(int, content[1].strip().split()))
                self.num_walls = wall_line[0]
                for i in range(self.num_walls):
                    x = wall_line[1 + 2*i]-1
                    y = wall_line[2 + 2*i]-1
                    self.walls.add((x, y))


                # Initialize num_boxes and boxes
                box_line = list(map(int, content[2].strip().split()))
                self.num_boxes = box_line[0]
                for i in range(self.num_boxes):
                    x = box_line[1 + 2*i]-1
                    y = box_line[2 + 2*i]-1
                    self.boxes.add((x, y))

                # Initialize num_storages and storages
                storage_line = list(map(int, content[3].strip().split()))
                self.num_storages = storage_line[0]
                for i in range(self.num_storages):
                    x = storage_line[1 + 2*i]-1
                    y = storage_line[2 + 2*i]-1
                    self.storages.add((x, y))

                # Intialize starting position
                x, y = map(int, content[4].strip().split())
                self.starting_pos = (x-1, y-1)

        except Exception as e:
            print(f"Error: {e}")

    def _initialize_game_board(self) -> None:
        """
        Initializes the game board as a 2D-Matrix

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

if __name__ == '__main__':
    board = Board(sys.argv[1])

    board.print()