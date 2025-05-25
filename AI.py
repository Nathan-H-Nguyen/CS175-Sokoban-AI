from collections import deque
from Board import *
import copy

class SokobanSolver:
    def __init__(self, board: Board):
        self.initial_board = board
        self.visited = set()

    def solve_BFS(self) -> list[str]:
        """
        Returns optimal path found by BFS.
        """
        start_state = (self.initial_board.player_pos, frozenset(self.initial_board.boxes))
        queue = deque([(start_state, [])])
        self.visited.add(start_state)

        while queue:
            (player_pos, boxes), path = queue.popleft()

            # Check win condition
            if boxes == self.initial_board.storages:
                return path

            for move in 'LRUD':
                result = self._simulate_move(player_pos, boxes, move)
                if result:
                    new_pos, new_boxes = result
                    # Check if not corner trapped
                    if not self._corner_trap(new_boxes):
                        # Check if already visited state
                        if (new_pos, new_boxes) not in self.visited:
                            self.visited.add((new_pos, new_boxes))
                            queue.append(((new_pos, new_boxes), path + [move]))

        return []
    
    def _get_new_position(self, pos: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Calculate new position based on direction."""
        x, y = pos
        direction_map = {
            'L': (x, y - 1),
            'R': (x, y + 1),
            'U': (x - 1, y),
            'D': (x + 1, y)
        }
        return direction_map[direction]
    
    def _simulate_move(self, player_pos, boxes, move):
        """
        Stateless simulation of move
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

    def _corner_trap(self, boxes):
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