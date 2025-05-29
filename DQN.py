from DQNEnv import DQNEnv
from Board import *
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import torch
from torch import nn
import torch.nn.functional as F
import os
from tqdm import trange
from typing import List

# Define Transition namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQN(nn.Module):
    def __init__(self, in_states: int, h1_nodes: int, h2_nodes: int):
        """
        Initializes the DQN object (Q-Network)

        Args:
            in_states (int): Number of input
            h1_nodes (int): Number of nodes in first hidden layer
            h2_nodes (int): Number of nodes in second hidden layer
        """
        super().__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes) # Input Layer
        self.fc2 = nn.Linear(h1_nodes, h2_nodes) # Hidden Layer
        self.fc3 = nn.Linear(h2_nodes, 4) # Output Layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Neural Network

        Args:
            x (torch.Tensor): Input representing state of environment (Flattened 1D vector)
        
        Returns:
        torch.Tensor: Output of raw Q-values after forward passing through Neural Network
        """
        x = F.relu(self.fc1(x)) # Run input through first layer (Input -> Hidden)
        x = F.relu(self.fc2(x)) # Run input through second layer (Hidden -> Hidden)
        return self.fc3(x) # Return Q-values for each action (Hidden -> Output)

class ReplayMemory:
    def __init__(self, maxlen: int):
        """
        Initializes the ReplayMemory object

        Args: maxlen (int): Max length of experiences to store
        """
        self.memory = deque(maxlen=maxlen)
    
    def append(self, transition: Transition) -> None:
        """
        Appends a Transition to the memory deque

        Args:
            transition (NamedTuple): A named tuple containing (state, action, next_state, reward, done), representing one step
                                        state: Previous state before taking action
                                        action: Action taken from state
                                        reward: Reward from action
                                        next_state: New state after taking action
                                        done: Indicates whether episode is done or not
        
        Returns:
            None
        """
        self.memory.append(transition)

    def sample(self, sample_size: int) -> List[Transition]:
        """
        Samples a random batch of transitions from the memory deque

        Args:
            sample_size (int): How many samples to get

        Returns:
            List[Transition]: List of randomly sampled transitions
        """
        return random.sample(self.memory, sample_size)

    def __len__(self) -> int:
        """
        Gets length of memory

        Returns:
            int: length of memory deque
        """
        return len(self.memory)
    
class Agent:
    def __init__(self, env: DQNEnv, learning_rate: float, discount_factor: float, epsilon: float, 
                in_states: int, h1_nodes: int, h2_nodes: int, maxlen: int):
        """
        Initializes the AI Agent with provided hyper parameters and DQN info

        Args:
            env (DQNEnv): Environment to act upon
            learning_rate (float): Alpha learning rate for optimizer
            discount_factor (float): Gamma discount factor for Q calculation
            epsilon (float): Epsilon-greedy action selection
            in_states (int): Size of input vector
            h1_nodes (int): Number of nodes in first hidden layer
            h2_nodes (int): Number of nodes in second hidden layer
            maxlen (int): Max length of ReplayMemory deque
        """
        # Initialize Environment
        self.env = env

        # Initialize Training Variables
        self.learning_rate = learning_rate # alpha
        self.discount_factor = discount_factor # gamma
        self.epsilon = epsilon # epsilon-greedy

        # Initialize DQN
        self.policy_dqn = DQN(in_states, h1_nodes, h2_nodes)
        self.target_dqn = DQN(in_states, h1_nodes, h2_nodes)

        # Initialize ReplayMemory
        self.memory = ReplayMemory(maxlen)

        # Initialize Optimizer
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr = learning_rate)

        # Initialize Step Counter
        self.step_counter = 0
    
    def select_action(self, state: torch.Tensor) -> int:
        """
        Selects an action from the given state, random or best move

        Args:
            state (torch.Tensor): 1D Tensor representing state of board/environment

        Returns:
            int: Action to take (0123 maps to LRUD respectively)
        """
        self.step_counter += 1 # Increment step

        # Epsilon-greedy policy to randomly decide on choosing random move or "best" move
        if random.random() < self.epsilon:
            return random.randint(0, 3) # 0123 => LRUD
        else:
            q_values = self.policy_dqn(state.unsqueeze(0)) # Get q values from output layer
            return torch.argmax(q_values).item() # Return best move
