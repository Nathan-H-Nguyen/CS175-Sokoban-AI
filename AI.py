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
                new_board = self._simulate_move(player_pos, boxes, move)
                if new_board and not new_board.box_corner_trap():
                    new_state = (new_board.player_pos, frozenset(new_board.boxes))
                    if new_state not in self.visited:
                        self.visited.add(new_state)
                        queue.append((new_state, path + [move]))

        return []

    def _simulate_move(self, player_pos, box_positions, move) -> Board:
        board_copy = copy.deepcopy(self.initial_board)
        board_copy.player_pos = player_pos
        board_copy.boxes = set(box_positions)
        board_copy._initialize_game_board()
        if board_copy.move(move):
            return board_copy
        return None


