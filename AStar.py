import heapq
from Board import *
import copy
from Typing import Tuple, Set, FrozenSet

class AStar:
    def __init__(self, board: Board):
        """
        Initializes the Astar object.

        Args:
            board (Board): Board object.

        Returns:
            None
        """
        self.initial_board = board
        self.visited = set()

    def solve(self) -> list[str]:
        """
        Solves puzzle using A*.

        Returns:
            list[str]: List representing each move found in the optimal path
        """
        iteration = 1

        # Initialize priority queue
        start_state = (self.initial_board.player_pos, frozenset(self.initial_board.boxes))
        f_score = self.heuristic(start_state[1])
        g_score = 0
        queue = []
        heapq.heappush(queue, (f_score, g_score, start_state, []))

        self.visited.add(start_state)

        while queue:
            f_score, g_score, (player_pos, boxes), path = heapq.heappop(queue)

            if boxes == self.initial_board.storages: # Check win condition
                print (f"\nTotal Iterations: {iteration}")
                return path

            for move in 'LRUD':
                result = self._simulate_move(player_pos, boxes, move)
                if result:
                    new_pos, new_boxes = result

                    if (new_pos, new_boxes) not in self.visited: # Check if already visited state
                        if not self._corner_trap(new_boxes): # Check if not corner trapped
                            iteration += 1
                            self.visited.add((new_pos, new_boxes))
                            new_g_score = g_score + 1
                            new_f_score = new_g_score + self.heuristic(new_boxes)
                            heapq.heappush(queue, (new_f_score, new_g_score, (new_pos, new_boxes), path + [move]))

        return []
    
    def heuristic(self, boxes: FrozenSet[Tuple[int, int]]) -> int:
        """
        Use Manhattan Distance from boxes to storage locations as the A* heuristic.

        Args:
            boxes (FrozenSet[Tuple[int, int]]): FrozenSet containing (x,y) tuples representing each box's position
        
        Returns:
            int: Manhattan Distance of all boxes to storage locations
        """
        manhattan_distance = 0
        for box in boxes:
            manhattan_distance += min((abs(box[0] - storage[0]) + abs(box[1] - storage[1])) for storage in self.initial_board.storages)
        
        return manhattan_distance

    def _get_new_position(self, pos: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """
        Calculate new position based on direction.

        Args:
            pos (Tuple[int, int]): (x,y) Tuple representing the current position
            direction (str): Direction to get new position
        
        Returns:
            Tuple[int, int]: (x,y) Tuple representing new position in direction
        """
        x, y = pos
        direction_map = {
            'L': (x, y - 1),
            'R': (x, y + 1),
            'U': (x - 1, y),
            'D': (x + 1, y)
        }
        return direction_map[direction]
    
    def _simulate_move(self, player_pos: Tuple[int, int], boxes: FrozenSet[Tuple[int, int]], move: str) -> Tuple[Tuple[int,int], FrozenSet[Tuple[int,int]]]:
        """
        Stateless simulation of move.

        Args:
            player_pos (Tuple[int, int]): (x,y) Tuple representing the player position
            Frozenboxes (Set[Tuple[int, int]]): Set of (x,y) Tuples representing the position of each box
            move (str): Direction to move

        Returns:
            Tuple[Tuple[int,int], FrozenSet[Tuple[int,int]]]:   Tuple containing the (x,y) coordinate of the player and 
                                                                a set of the (x,y) coordinates of boxes
        """
        new_pos = self._get_new_position(player_pos, move)

        # If direction is wall cant move
        if new_pos in self.initial_board.walls:
            return None
        
        # If box in the way, attempt to push box
        if new_pos in boxes:
            x, y = new_pos
            pos_behind_box = self._get_new_position(new_pos, move)

            # Position behind box is not empty, then cant push/move
            if pos_behind_box in self.initial_board.walls or pos_behind_box in boxes:
                return None

            # Push Box
            new_boxes = set(boxes)
            new_boxes.remove(new_pos) # Remove box from current space
            new_boxes.add(pos_behind_box) # Move box to space behind
        
            return (new_pos, frozenset(new_boxes))
        
        # No box or wall, empty space

        return (new_pos, frozenset(boxes))

    def _corner_trap(self, boxes: FrozenSet[Tuple[int,int]]) -> bool:
        """
        Checks if any box is trapped in a corner and not on a storage location.

        Returns:
            bool: True if a box is trapped in a corner, else False
        """
        for box in boxes:
            # If box in storage location skip, no need to check if trapped
            if box in self.initial_board.storages:
                continue

            left = self._get_new_position(box, 'L')
            right = self._get_new_position(box, 'R')
            up = self._get_new_position(box, 'U')
            down = self._get_new_position(box, 'D')

            if up in self.initial_board.walls and right in self.initial_board.walls:  # Top right corner
                return True
            elif right in self.initial_board.walls and down in self.initial_board.walls:  # Bottom right corner
                return True
            elif down in self.initial_board.walls and left in self.initial_board.walls:  # Bottom left corner
                return True
            elif left in self.initial_board.walls and up in self.initial_board.walls:  # Top left corner
                return True

        return False