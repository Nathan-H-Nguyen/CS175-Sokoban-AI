from Board import *
import torch
from typing import Tuple, FrozenSet, Dict

class DQNEnv:
    def __init__(self, board: Board, max_size: int, step_limit: int):
        """
        Initializes the environment wrapper that will be used for DQN.

        Args:
            board (Board): Board object.
            max_size (int): Max size of boards in training
            step_limit (int): Number of steps/moves we allow before stopping episode

        Returns:
            None
        """
        self.board = board
        self.max_size = max_size
        self.step_count = 0
        self.step_limit = step_limit
        self.invalid_move_streak = 0

        self.reset()
    
    def reset(self) -> torch.Tensor:
        """
        Resets the environment and returns the state.

        Returns:
            torch.Tensor: Flattened 1D vector state representation of board
        """
        self.board.reset()
        self.step_count = 0
        self.invalid_move_streak = 0
        return self._get_state()
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict[str, object]]:
        """
        Performs an action in the environment and returns the relevant results

        Args:
            action (int): Action to be made on the board (0123 => LRUD respectively)
        
        Returns:
            Tuple[torchTensor, float, bool, Dict[str, object]]: torch.Tensor
                                                                    Flattened state of board
                                                                float
                                                                    Reward after action
                                                                bool
                                                                    Whether game is done or not
                                                                Dict[str, object]
                                                                    Info regarding move

        """
        move_mapping = {0: 'L',
                        1: 'R',
                        2: 'U',
                        3: 'D'}
        
        # Get move and old player and box positions
        move = move_mapping[action]
        old_player = tuple(self.board.player_pos)
        old_boxes = frozenset(self.board.boxes)

        # Make move and increment count
        self.board.move(move)
        self.step_count += 1

        # Get new player and box positions
        new_player = set(self.board.player_pos)
        new_boxes = frozenset(self.board.boxes)

        # Calculate reward after making move
        reward = self._calculate_reward(old_player, new_player, old_boxes, new_boxes)

        # Get current state
        state = self._get_state()

        # Log info if needed
        info = {"action": (action, move),
                "step_count": self.step_count,
                "moved": 1 if old_player != new_player else 0,
                "invalid_move": 1 if old_player == new_player else 0,
                "pushed_box": 1 if old_boxes != new_boxes else 0,
                "reward": reward,
                "done": self.board.is_win() or self.board.box_corner_trap() or self.board.adjacent_box_trap() or self.board.unpushable_boxes_trap() or self.invalid_move_streak >= 10 or self.step_count >= self.step_limit,
                "win": self.board.is_win(),
                "corner_trap": self.board.box_corner_trap(),
                "adjacent_box_trap": self.board.adjacent_box_trap(),
                "unpushable_box_trap": self.board.unpushable_box_trap(),
                "invalid_move_streak": self.invalid_move_streak >= 10,
                "exceed_step_limit": self.step_count >= self.step_limit,
                "player_pos": new_player, # Maybe use when visualizing?
                "boxes_pos": new_boxes, # Maybe use when visualizing?
                "num_boxes_on_storage": sum(box in self.board.storages for box in new_boxes),
                "num_boxes_not_on_storage": sum(box not in self.board.storages for box in new_boxes),
        }

        return (state, reward, info['done'], info)
    
    def _get_state(self) -> torch.Tensor:
        """
        Gets the current state of the board in an "ingestable" way for DQN.
        The state will have 6 different values:
            0 - Empty Squares
            1 - Player
            2 - Boxes
            3 - Storage Locations
            4 - Boxes on Storage Locations
            5 - Walls
        
        Returns:
            torch.Tensor: Flattened 1D vector to represent the state
        """
        # Create np 2D Matrix for board state
        state = torch.zeros((self.board.rows, self.board.cols), dtype=torch.float32)

        # Set Player
        player_x, player_y = self.board.player_pos
        state[player_x, player_y] = 1

        # Get unassigned Boxes and unassigned Storages
        unassigned_boxes = {box for box in self.board.boxes if box not in self.board.storages}
        unassigned_storages = {storage for storage in self.board.storages if storage not in self.board.boxes}

        # Set Boxes not on Storage Locations
        for box_x, box_y in unassigned_boxes:
            state[box_x, box_y] = 2
        
        # Set empty Storage Locations
        for storage_x, storage_y in unassigned_storages:
            state[storage_x, storage_y] = 3
        
        # Get assigned Boxes
        assigned_boxes = self.board.boxes - unassigned_boxes

        # Set Boxes on Storage Locations
        for box_x, box_y in assigned_boxes:
            state[box_x, box_y] = 4
        
        # Set Walls
        for wall_x, wall_y in self.board.walls:
            state[wall_x, wall_y] = 5
        
        # Flatten 2D Matrix into 1D vector for DQN input
        flattened_state = state.flatten()
        
        # Zero Pad if needed
        if flattened_state.numel() < self.max_size:
            padding = torch.zeros(self.max_size - flattened_state.numel(), dtype=flattened_state.dtype)
            flattened_state = torch.cat([flattened_state, padding])
        
        return flattened_state

    def _calculate_reward(self, old_player: Tuple[int, int], new_player: Tuple[int, int], old_boxes: FrozenSet[Tuple[int, int]], new_boxes: FrozenSet[Tuple[int, int]]) -> float:
        """
        Calculates Reward after making a move.

        Args:
            old_player (Tuple[int, int]): (x, y) coordinates of player's old position
            new_player (Tuple[int, int]): (x, y) coordinates of player's new position
            old_boxes (FrozenSet[Tuple[int, int]]): Set of (x, y) Tuples representing position of boxes before
                                                    move was made
            new_boxes (FrozenSet[Tuple[int, int]]): Set of (x, y) Tuples representing position of boxes after
                                                    move was made
        
        Returns:
            float: Reward after move.
        """
        reward = 0

        # FAT Reward for winning
        if new_boxes == self.board.storages:
            return 100.0
        
        # Penalty for deadlock -20
        if self.board.box_corner_trap() or self.board.adjacent_box_trap() or self.board.unpushable_boxes_trap():
            return -20.0
        
        # Penalty for invalid move
        # i.e. tried to move in wall or push box that cant be pushed
        if old_player == new_player:
            self.invalid_move_streak += 1
            return -10.0
        else:
            self.invalid_move_streak = 0

        # Reward for pushing Box onto Storage Location
        for box in new_boxes:
            if box not in old_boxes and box in self.board.storages:
                reward += 10.0
        
        # Penalty for pushing Box off Storage Location
        # Less than pushed onto bc sometimes you gotta push off to solve
        for box in old_boxes:
            if box not in new_boxes and box in self.board.storages:
                reward -= 2.5
        
        # Reward for pushing box
        if old_boxes != new_boxes:
            reward += 1.0

        # Penalty for moving to encourage solution with less moves
        reward -= 0.10

        return reward
