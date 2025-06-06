from Board import Board
from typing import Tuple, Set, FrozenSet

class Solver:
    def __init__(self, board: Board):
        self.initial_board = board
        self.visited = set()

    def solve(self):
        raise NotImplementedError()


    def _expand_moves(self, player_pos: Tuple[int, int], boxes: FrozenSet[Tuple[int, int]]) -> Tuple[Tuple[int, int], FrozenSet[Tuple[int, int]]]:
        """
        Expand moves in all directions and yield results

        Args:
            player_pos (Tuple[int, int]): (x,y) Tuple representing the current position
            boxes (FrozenSet[Tuple[int, int]]): current boxes in state

        Returns:
            player_pos, boxes: returns updated arguments
        """
        for move in 'LRUD':
            result = self._simulate_move(player_pos, boxes, move)
            if result:
                new_pos, new_boxes = result
                if (new_pos, new_boxes) not in self.visited and not self._deadlock(new_boxes):  # Check if already visited state
                    yield new_pos, new_boxes, move


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

    def _simulate_move(self, player_pos: Tuple[int, int], boxes: FrozenSet[Tuple[int, int]],
                       move: str) -> Tuple[Tuple[int, int], FrozenSet[Tuple[int, int]]]:
        """
        Stateless simulation of move.

        Args:
            player_pos (Tuple[int, int]): (x,y) Tuple representing the player position
            boxes (Set[Tuple[int, int]]): Set of (x,y) Tuples representing the position of each box
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
            new_boxes.remove(new_pos)  # Remove box from current space
            new_boxes.add(pos_behind_box)  # Move box to space behind

            return new_pos, frozenset(new_boxes)

        # No box or wall, empty space

        return new_pos, frozenset(boxes)


    def _deadlock(self, boxes: FrozenSet[Tuple[int, int]]) -> bool:
        return self._corner_trap(boxes) or self._adjacent_box_trap(boxes) or self._unpushable_boxes_trap(boxes)

    def _corner_trap(self, boxes: FrozenSet[Tuple[int, int]]) -> bool:
        """
        Checks if any box is trapped in a corner and not on a storage location.

        Args:
            boxes (FrozenSet[Tuple[int,int]]): Set of (x,y) Tuples representing the position of each box

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

    # Deadlock Check
    def _adjacent_box_trap(self, boxes: FrozenSet[Tuple[int, int]]) -> bool:
        """
        Checks if any adjacent boxes are trapped.

        Args:
            boxes (FrozenSet[Tuple[int,int]]): Set of (x,y) Tuples representing the position of each box

        Returns:
            bool: True if a box is trapped, else False
        """
        for box in boxes:
            # If box in storage location skip, no need to check if trapped
            if box in self.initial_board.storages:
                continue

            # Get squares to the LRUD of box
            left = self._get_new_position(box, 'L')
            right = self._get_new_position(box, 'R')
            up = self._get_new_position(box, 'U')
            down = self._get_new_position(box, 'D')

            # If box is along a wall (left or right), check if there's a box above or below it
            # And if that box is along the same wall
            if left in self.initial_board.walls:
                if up in boxes and not self._recursive_can_push_box(boxes, up, set()):
                    return True
                elif down in boxes and not self._recursive_can_push_box(boxes, down, set()):
                    return True
            if right in self.initial_board.walls:
                if up in boxes and not self._recursive_can_push_box(boxes, up, set()):
                    return True
                elif down in boxes and not self._recursive_can_push_box(boxes, down, set()):
                    return True

            # If box is along a wall (above or below), check if theres a box left or right of it
            # And if that box is along the same wall
            if up in self.initial_board.walls:
                if left in boxes and not self._recursive_can_push_box(boxes, left, set()):
                    return True
                elif right in boxes and not self._recursive_can_push_box(boxes, right, set()):
                    return True
            if down in self.initial_board.walls:
                if left in boxes and not self._recursive_can_push_box(boxes, left, set()):
                    return True
                elif right in boxes and not self._recursive_can_push_box(boxes, right, set()):
                    return True

        return False

    # Deadlock Check
    def _unpushable_boxes_trap(self, boxes) -> bool:
        """
        Checks if all boxes not on a storage are unpushable

        Returns:
            bool: True if all boxes not on storages are unpushable, otherwise False
        """
        if boxes == self.initial_board.storages:
            return False
        
        for box in boxes:
            # If box is on a storage ignore it
            if box in self.initial_board.storages:
                continue

            if self._can_push_box(boxes, box):
                return False

        return True

    def _can_push_box(self, boxes: FrozenSet[Tuple[int, int]], box: Tuple[int, int]) -> bool:
        """
        Checks if a box can be pushed in ANY direction

        Args:
            box (Tuple[int, int]): box to check if pushable

        Returns:
            bool: True if can be pushed, False otherwise
        """
        opposite_direction = {
            'L': 'R',
            'R': 'L',
            'U': 'D',
            'D': 'U',
        }

        # Check if box can be pushed from any direction
        for direction in 'LRUD':
            push_to = self._get_new_position(box, direction)  # Get square in front of box (push to)
            push_from = self._get_new_position(box, opposite_direction[
                direction])  # Get square behind box (push from)

            # If push to square and push from square are empty, we can push!
            if push_to not in self.initial_board.walls and push_to not in boxes:
                if push_from not in self.initial_board.walls and push_from not in boxes:
                    return True

        return False


    def _recursive_can_push_box(self, boxes: FrozenSet[Tuple[int, int]], box: Tuple[int, int], visited: Set[Tuple[int, int]]) -> bool:
        """
        Recurisvely check if box and its neighbors (if boxes) can be pushed

        Args:
            box (Tuple[int, int]): Current box we are checking
            visited (Set[Tuple[int, int]]): Boxes we've already visited

        Returns:
            bool: True if any box and/or its neighbors can be pushed, False otherwise
        """

        #wrapper recursive function to avoid duplicating boxes (expensive)
        def recursiveFunc(box: Tuple[int, int], visited: Set[Tuple[int, int]]) -> bool:
            # Base case, if box already visited skip
            if box in visited:
                return False
            visited.add(box)

            # Base case, if we can push box return true
            if self._can_push_box(boxes, box):
                return True

            # Can't push box, check boxes around and see if they can be pushed
            for direction in 'LRUD':
                adjacent = self._get_new_position(box, direction)
                if adjacent in self.initial_board.walls:
                    continue

                if adjacent in self.initial_board.boxes:
                    if self._recursive_can_push_box(boxes, adjacent, visited):
                        return True

            return False

        return recursiveFunc(box, visited)
