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
        self.ELEMENTS = {
            'wall': '#',
            'box': '$',
            'storage': '.',
            'player': '@',
            'empty': ' '
        }

        self.file = file  # Input file

        self.rows = 0  # Number of rows in the board
        self.cols = 0  # Number of columns in the board
        self.board = []  # 2D-Matrix for board itself

        self.num_walls = 0  # Total number of walls
        self.walls = set()  # Set containing tuples() of the (x,y) coordinates

        self.num_boxes = 0  # Total number of boxes
        self.boxes = set()  # Set containing tuples() of the (x,y) coordinates

        self.num_storages = 0  # Total number of storage locations
        self.storages = set()  # Set containing tuples() of the (x,y) coordinates

        self.starting_pos = tuple()  # Starting (x,y) location
        self.player_pos = tuple()  # Current player (x,y) location

        # Initialize data members and board
        self._initialize_data_members()
        self._initialize_game_board()

    def _initialize_data_members(self) -> None:
        """
        Parses input file and initializes all data members other than board itself.

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
                    x = wall_line[1 + 2 * i] - 1
                    y = wall_line[2 + 2 * i] - 1
                    self.walls.add((x, y))

                # Initialize num_boxes and boxes
                box_line = list(map(int, content[2].strip().split()))
                self.num_boxes = box_line[0]
                for i in range(self.num_boxes):
                    x = box_line[1 + 2 * i] - 1
                    y = box_line[2 + 2 * i] - 1
                    self.boxes.add((x, y))

                # Initialize num_storages and storages
                storage_line = list(map(int, content[3].strip().split()))
                self.num_storages = storage_line[0]
                for i in range(self.num_storages):
                    x = storage_line[1 + 2 * i] - 1
                    y = storage_line[2 + 2 * i] - 1
                    self.storages.add((x, y))

                # Initalize starting position
                x, y = map(int, content[4].strip().split())
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
        self.board = [[' '] * self.cols for row in range(self.rows)]

        # Place Walls
        for x, y in self.walls:
            self.board[x][y] = self.ELEMENTS['wall']

        # Place Boxes
        for x, y in self.boxes:
            self.board[x][y] = self.ELEMENTS['box']

        # Place Storage Locations
        for x, y in self.storages:
            self.board[x][y] = self.ELEMENTS['storage']

        # Place Player
        self.board[self.starting_pos[0]][self.starting_pos[1]] = self.ELEMENTS['player']

    ########### PUBLIC METHODS ##########
    def move(self, direction: str) -> bool:
        """
        Attempts to move the player in the specified direction.

        Returns:
            bool: True if player moved, False otherwise
        """
        """Attempt to move the player in the given direction."""
        new_pos = self._get_new_position(self.player_pos, direction)

        if not self._is_valid_move(new_pos):
            return False

        if new_pos in self.boxes:
            if not self._push_box(new_pos, direction):
                return False

        # Update player position
        self._update_position(self.player_pos, new_pos)
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
    def _get_new_position(self, pos: Tuple[int, int], direction: str) -> Tuple[int, int]:
        print("getnewposition")
        """Calculate new position based on direction."""
        x, y = pos
        direction_map = {
            'L': (x, y - 1),
            'R': (x, y + 1),
            'U': (x - 1, y),
            'D': (x + 1, y)
        }
        return direction_map[direction]

    def _is_valid_move(self, pos: Tuple[int, int]) -> bool:
        print("isvalid")
        """Check if a position is valid for movement."""
        return pos not in self.walls

    def _push_box(self, box_pos: Tuple[int, int], direction: str) -> bool:
        print("pushbox")
        """Attempt to push a box in the given direction."""
        new_box_pos = self._get_new_position(box_pos, direction)

        if not self._is_valid_move(new_box_pos):
            return False

        # Update box position
        self.boxes.remove(box_pos)
        self.boxes.add(new_box_pos)

        # Update board
        self._update_box_position(box_pos, new_box_pos)
        return True

    def _update_position(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int]) -> None:
        print("update pos")
        """Update player position on the board."""
        x, y = old_pos
        self.board[x][y] = (self.ELEMENTS['storage']
                            if old_pos in self.storages
                            else self.ELEMENTS['empty'])

        self.player_pos = new_pos
        x, y = new_pos
        self.board[x][y] = self.ELEMENTS['player']

    def _update_box_position(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int]) -> None:
        print("update box pos")
        """Update box position on the board."""
        x, y = old_pos
        self.board[x][y] = (self.ELEMENTS['storage']
                            if old_pos in self.storages
                            else self.ELEMENTS['empty'])

        x, y = new_pos
        self.board[x][y] = self.ELEMENTS['box']