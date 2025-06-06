import heapq
from collections import deque
from Board import *
import copy
from typing import Tuple, Set, FrozenSet
from solver import Solver
from scipy.optimize import linear_sum_assignment


class AStar(Solver):
    def __init__(self, initial_board):
        super().__init__(initial_board)
        self.hungarian_cache = dict()

    def solve(self) -> list[str]:
        """
        Solves puzzle using A*.

        Returns:
            list[str]: List representing each move found in the optimal path
        """
        iteration = 1

        # Initialize priority queue
        start_state = (self.initial_board.player_pos, frozenset(self.initial_board.boxes))
        f_score = self.manhattan_heuristic(start_state[1])
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
                new_f_score = new_g_score + self.manhattan_heuristic(new_boxes)
                heapq.heappush(queue, (new_f_score, new_g_score, (new_pos, new_boxes), path + [move]))

        return []
    
    def manhattan_heuristic(self, boxes: FrozenSet[Tuple[int, int]]) -> int:
        """
        Use Manhattan Distance from boxes to storage locations as the A* heuristic.

        Args:
            boxes (FrozenSet[Tuple[int, int]]): FrozenSet containing (x,y) tuples representing each box's position
        
        Returns:
            int: Manhattan Distance of all boxes to storage locations
        """
        return sum(min((abs(box[0] - storage[0]) + abs(box[1] - storage[1])) for storage in self.initial_board.storages) for box in boxes)

    def hungarian_heuristic(self, boxes: FrozenSet[Tuple[int, int]]) -> int:
        if boxes in self.hungarian_cache:
            return self.hungarian_cache[boxes]
        
        cost_matrix = self._create_cost_matrix(boxes)

        # Hungarian!
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        cost = sum(cost_matrix[row][col] for row,col in zip(row_indices, col_indices))
        self.hungarian_cache[boxes] = cost
        return cost

    def _create_cost_matrix(self, boxes: FrozenSet[Tuple[int, int]]):
        # Store boxes and storages as a list for indexing
        boxes = list(boxes)
        storages = list(self.initial_board.storages)


        cost_matrix = [[float('inf')] * self.initial_board.num_storages for box in range(self.initial_board.num_boxes)]

        # Start from storage and reverse BFS to boxes
        for col,storage in enumerate(storages):
            queue = deque([(storage, 0)])
            visited = {storage}
            distance = {}

            # Run BFS from storage and get all reachables tiles (Not walls)
            while queue:
                pos, dist = queue.popleft()
                distance[pos] = dist

                # Expand
                for direction in 'LRUD':
                    new_pos = self.initial_board._get_new_position(pos, direction)

                    # If move is invalid (wall) or already visited skip
                    if new_pos in self.initial_board.walls or new_pos in visited:
                        continue

                    visited.add(new_pos)
                    queue.append((new_pos, dist+1))

            # Check what boxes were reachable from this storage, and store its distance cost
            for row,box in enumerate(boxes):
                cost_matrix[row][col] = distance[box]
        
        return cost_matrix
