import heapq
from Board import *
import copy
from typing import Tuple, Set, FrozenSet
from solver import Solver


class AStar(Solver):
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

            for new_pos, new_boxes, move in self._expand_moves(player_pos, boxes):
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
        return sum(min((abs(box[0] - storage[0]) + abs(box[1] - storage[1])) for storage in self.initial_board.storages) for box in boxes)