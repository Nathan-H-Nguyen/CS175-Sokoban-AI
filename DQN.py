import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import os
from tqdm import trange

class DQN(nn.Module):
    def __init__(self, in_states: int, h1_nodes: int, h2_nodes: int, out_actions: int):
        """
        Initializes the Q-Network.

        Args:
            in_states (int): Number of input
            h1_nodes (int): Number of nodes in first hidden layer
            h2_nodes (int): Number of nodes in second hidden layer
            out_actions (int): Number of output
        """
        super().__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes) # Input Layer
        self.fc2 = nn.Linear(h1_nodes, h2_nodes) # Hidden Layer
        self.fc3 = nn.Linear(h2_nodes, out_actions) # Output Layer

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
