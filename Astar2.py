from collections import deque
# from heapq import heappush, heappop
from Board import Board
import queue
from typing import Tuple
import time

class State:
    def __init__(self, player: Tuple[int, int], boxes: set, goals: set, direction: str, n_moves: int):
        self.player = player
        self.boxes = frozenset(boxes)
        self.goals = frozenset(goals)
        self.direction = direction
        self.n_moves = n_moves

    def __lt__(self, other) -> bool:
        return self.n_moves + self.heuristic() < other.n_moves + other.heuristic()

    def __eq__(self, other):
        return self.player == other.player and self.boxes == other.boxes and self.direction == other.direction

    def __hash__(self):
        return hash((self.player, self.boxes, self.direction))

    def __repr__(self):
        return f"player: {self.player}, value: {self.n_moves + self.heuristic()}, boxes: {self.boxes}"


    @staticmethod
    def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def heuristic(self):
        return sum(min(State.manhattan_distance(box, goal) for goal in self.goals) for box in self.boxes)



class AstarSolver:
    def __init__(self, board: Board):
        self.board = board
        self._direction_map = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
        self._direction_map_inverse = {(0, -1): 'L',  (0, 1): 'R', (-1, 0): 'U', (1, 0): 'D'}

    def solveOld(self):
        state = State(self.board.player_pos, self.board.boxes, self.board.storages, 0)
        visited = set()
        pq = queue.PriorityQueue()
        pq.put(state)

        while not pq.empty():
            state = pq.get()
            #print(f"current state: {state}")

            if state.boxes == frozenset({(2,3), (3,4)}):
                print("Done")
                break
            visited.add(state)
            # print(visited)

            for move in 'LRUD':
                simulated = self._simulate_move(state.player, state.boxes, move)
                if simulated is None:
                    continue
                new_pos, boxes = simulated
                new_state = State(new_pos, boxes, self.board.storages, state.n_moves + 1)
                if new_state not in visited and not self._corner_trap(boxes):
                    pq.put(new_state)
            #print(pq.queue)
            #input()




    def solve(self):
        #state = State(self.board.player_pos, self.board.boxes, self.board.storages, 0)
        reachable = self._reachable_squares(self.board.player_pos, self.board.boxes)
        boxPositions = self._box_push_positions(reachable, self.board.boxes)
        pq = queue.PriorityQueue()
        for position, direction in boxPositions.items():
            result = self._simulate_move(position, self.board.boxes, direction)
            if result is None: continue
            new_pos, boxes = result
            state = State(new_pos, boxes, self.board.storages, direction, self._uncover_path(position, reachable)[1])
            #print(f"pos: {position}, len: {self._uncover_path(position, reachable)[1]}, h: {state.heuristic()}")
            pq.put(state)
        visited = set()

        while not pq.empty():
            state = pq.get()
            #print(f"current state: {state}")
            visited.add(state)

            if state.boxes == self.board.storages:
                print("Done")
                return

            #simulated = self._simulate_move(state.player, state.boxes, state.direction)
            # if simulated is None:
            #     continue
            # new_pos, new_boxes = simulated
            # if self._corner_trap(new_boxes):
            #     continue
            reachable = self._reachable_squares(state.player, state.boxes)
            boxPositions = self._box_push_positions(reachable, state.boxes)
            for position, direction in boxPositions.items():
                result = self._simulate_move(position, state.boxes, direction)
                if result is None: continue
                new_pos, boxes = result
                new_state = State(new_pos, boxes, self.board.storages, direction, state.n_moves + self._uncover_path(position, reachable)[1])
                if new_state not in visited:
                    #print(f"pos: {position}, len: {state.n_moves + self._uncover_path(position, reachable)[1]}, h: {new_state.heuristic()}")
                    pq.put(new_state)
            #print(pq.queue)
            #input()




    def _get_new_position(self, pos: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Calculate new position based on direction."""
        dX, dY = self._direction_map[direction]
        return pos[0] + dX, pos[1] + dY

    def _simulate_move(self, player_pos, boxes, move):
        """
        Stateless simulation of move
        """
        new_pos = self._get_new_position(player_pos, move)

        # If direction is wall cant move
        if new_pos in self.board.walls:
            return None

        # If box in the way, attempt to push box
        if new_pos in boxes:
            x, y = new_pos
            pos_behind_box = self._get_new_position(new_pos, move)

            # Position behind box is not empty, then cant push/move
            if pos_behind_box in self.board.walls or pos_behind_box in boxes:
                return None

            # Push Box
            new_boxes = set(boxes)
            new_boxes.remove(new_pos)  # Remove box from current space
            new_boxes.add(pos_behind_box)  # Move box to space behind

            return new_pos, new_boxes

        # No box or wall, empty space

        return new_pos, boxes


    def _corner_trap(self, boxes):
        """
        Checks if any box is trapped in a corner and not on a storage location.

        Returns:
            bool: True if a box is trapped in a corner, else False
        """
        for box in boxes:
            # If box in storage location skip, no need to check if trapped
            if box in self.board.storages:
                continue

            left = self._get_new_position(box, 'L')
            right = self._get_new_position(box, 'R')
            up = self._get_new_position(box, 'U')
            down = self._get_new_position(box, 'D')

            if up in self.board.walls and right in self.board.walls:  # Top right corner
                return True
            elif right in self.board.walls and down in self.board.walls:  # Bottom right corner
                return True
            elif down in self.board.walls and left in self.board.walls:  # Bottom left corner
                return True
            elif left in self.board.walls and up in self.board.walls:  # Top left corner
                return True

        return False

    def _uncover_path(self, pos: Tuple[int, int], reachable: dict[Tuple[int, int], Tuple[int, int]]):
        """
        Params:
            :param pos: Position found in reachable.
            :param reachable: dict of reachable squares as found through _reachable_squares
        Returns:
            path from player position to pos, and the number of steps taken
        """
        if not pos in reachable:
            return None
        path = [pos]
        while reachable[path[-1]]:
            path.append(reachable[path[-1]])
        path.reverse()
        return path, len(path)

    def _box_push_positions(self, reachable: dict[Tuple[int, int], Tuple[int, int]], boxes: set) -> dict[Tuple[int, int], str]:
        """
        Finds all positions to push a box from.
        Returns:
            Positions behind boxes to push, with respective directions to box positions.
        """
        positions = dict()
        for bX, bY in boxes:
            for direction, (dX, dY) in self._direction_map.items():
                behindBox = (bX - dX, bY - dY)
                frontBox = (bX + dX, bY + dY)
                if behindBox in reachable and self._empty_square(*frontBox, boxes):
                    positions[behindBox] = self._direction_map_inverse[(dX, dY)]
        return positions

    def _reachable_squares(self, player_pos: Tuple[int, int], boxes: set):
        """
        Performs BFS to find the shortest path from player_pos to all reachable squares.
        :param player_pos: tuple containing (x,y) coordinates of player position
        :param boxes: set of box coordinates
        :return: dictionary containing reachable squares as keys, and their shortest path parent as value
        """
        reachable = dict()
        queue = deque()
        queue.append((player_pos[0], player_pos[1]))
        reachable[player_pos[0], player_pos[1]] = None

        while queue:
            x, y = queue.popleft()
            for dX, dY in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                newX, newY = x + dX, y + dY
                if (newX, newY) not in reachable and self._empty_square(newX, newY, boxes):
                    reachable[newX, newY] = (x, y)
                    queue.append((newX, newY))
        return reachable


    def _empty_square(self, x, y, boxes):
        return 1 <= x <= self.board.rows and 1 <= y <= self.board.cols and (x, y) not in self.board.walls and (x, y) not in boxes
