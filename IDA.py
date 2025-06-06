from Astar import AStar
from Board import Board

from collections import deque

class IDA(AStar):
    def solve(self):
        bound = self.hungarian_heuristic(frozenset(self.initial_board.boxes))
        state = (self.initial_board.player_pos, frozenset(self.initial_board.boxes))
        path_stack = deque()
        path_stack.append(state)
        path_moves = deque()
        self.iteration = 1

        while True:
            transposition = {}
            t = self._search(path_stack, path_moves, set(), 0, bound, transposition)
            if t == 0:
                print(f"\nTotal Iterations: {self.iteration}")
                return path_moves
            if t == float('inf'):
                return None
            bound = t
            self.iteration += 1



    def _search(self, path_stack: deque, path_moves: deque, path_visited: set, g, bound, transposition):
        player_pos, boxes = path_stack[-1]
        state = (player_pos, boxes)
        f = g + self.hungarian_heuristic(boxes)
        if f > bound:
            return f

        if boxes == self.initial_board.storages:
            return 0
        
        # If we've already visited/seen this state at a lower cost then stop search
        if state in transposition and g >= transposition[state]:
            return float('inf')
        transposition[state] = g

        minimum = float('inf')
        for new_pos, new_boxes, move in self._expand_moves(player_pos, boxes):
            if (new_pos, new_boxes) not in path_visited:
                path_stack.append((new_pos, new_boxes))
                path_visited.add((new_pos, new_boxes))
                path_moves.append(move)
                t = self._search(path_stack, path_moves, path_visited, g + 1, bound, transposition)
                if t == 0:
                    return t
                if t < minimum:
                    minimum = t

                path_stack.pop()
                path_moves.pop()
                path_visited.remove((new_pos, new_boxes))

        return minimum



