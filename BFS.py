from collections import deque
from Board import *
from typing import Tuple, Set, FrozenSet
from solver import Solver

class BFS(Solver):

    def solve(self) -> list[str]:
        """
        Solves puzzle using BFS.

        Returns:
            list[str]: List representing each move found in the optimal path
        """
        iteration = 1
        start_state = (self.initial_board.player_pos, frozenset(self.initial_board.boxes))
        queue = deque([(start_state, [])])
        self.visited.add(start_state)

        while queue:
            (player_pos, boxes), path = queue.popleft()

            if boxes == self.initial_board.storages: # Check win condition
                print (f"Total Iterations: {iteration}")
                return path

            for move in 'LRUD':
                result = self._simulate_move(player_pos, boxes, move)
                if result:
                    new_pos, new_boxes = result
                    if (new_pos, new_boxes) not in self.visited: # Check if already visited state
                        if not self._corner_trap(new_boxes): # Check if not corner trapped
                            iteration += 1
                            self.visited.add((new_pos, new_boxes))
                            queue.append(((new_pos, new_boxes), path + [move]))

        return []
    
