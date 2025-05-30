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
                batch_size: int, sync_rate: int, in_states: int, h1_nodes: int, h2_nodes: int, 
                maxlen: int):
        """
        Initializes the AI Agent with provided hyper parameters and DQN info

        Args:
            env (DQNEnv): Environment to act upon
            learning_rate (float): Alpha learning rate for optimizer
            discount_factor (float): Gamma discount factor for Q calculation
            epsilon (float): Epsilon-greedy action selection
            batch_size (int): Size of sample batch from ReplayMemory
            sync_rate (int): When to sync policy_dqn and target_dqn
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
        self.batch_size = batch_size
        self.sync_rate = sync_rate

        # Initialize DQN
        self.policy_dqn = DQN(in_states, h1_nodes, h2_nodes)
        self.target_dqn = DQN(in_states, h1_nodes, h2_nodes)
        # Load previous training weights
        if os.path.exists("dqn.pt"):
            self.policy_dqn.load_state_dict(torch.load("dqn.pt"))
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # Initialize ReplayMemory
        self.memory = ReplayMemory(maxlen)

        # Initialize Optimizer and Loss Function
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr = learning_rate)
        self.loss_fn = nn.MSELoss()

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
        
    def train(self, episodes: int) -> None:
        """
        Trains the Agent on the level episodes amount of times, changing weights and optimizing as needed

        Args:
            episodes (int): Number of episodes to train on

        Returns:
            None
        """
        rewards_per_episode = np.zeros(episodes) # List to keep track of rewards collected per episode
        epsilon_history = np.zeros(episodes) # List to keep track of epsilon decay

        # Train for x episodes
        for i in range(episodes):
            state = self.env.reset()
            done = False # True when puzzle is solved or exceeded step limit (in DNQEnv)
            accumulated_reward = 0

            # While puzzle hasnt been solved and havent exceeded step limit
            while (not done):
                # Select Action for current state
                action = self.select_action(state)

                # Make Action, get results
                new_state, reward, done, info = self.env.step(action)
                accumulated_reward += reward

                # Append Transition to Memory
                transition = Transition(state, action, reward, new_state, done)
                self.memory.append(transition)

                # Update State
                state = new_state
            
            rewards_per_episode[i] = accumulated_reward

            # Grab Sample Batch from Memory and Optimize
            if len(self.memory) >= self.batch_size:
                batch = self.memory.sample(self.batch_size)
                self.optimize(batch)

                # Sync policy_dqn and target_dqn
                if self.step_counter >= self.sync_rate:
                    self.step_counter = 0
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
            
            # Decay Epsilon
            self.epsilon = max(self.epsilon - (1/episodes), 0)
            epsilon_history[i] = self.epsilon
        
        # Save Policy
        torch.save(self.policy_dqn.state_dict(), 'dqn.pt')

        # Save Results Figure
        plt.figure(figsize=(12, 5))

        # Track accumulater reward per episode
        # Episode vs Accumulated Reward
        plt.subplot(1, 2, 1)
        plt.plot(rewards_per_episode)
        plt.title("Accumulated Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Accumulated Reward")

        # Track epsilon decay per episode
        # Episode vs Epsilon
        plt.subplot(1, 2, 2)
        plt.plot(epsilon_history)
        plt.title("Epsilon Decay per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")

        plt.tight_layout()
        plt.savefig("dqn_training_results.png")
        plt.close()

    def optimize(self, batch: List[Transition]) -> None:
        """
        Optimizes policy dqn using batch from ReplayMemory

        Args:
            batch (List[Transition]): List of transitions sampled from memory
        
        Returns: None
        """
        current_q_list = []
        target_q_list = []

        for state, action, reward, next_state, done in batch:
            # Get Q-values from policy for current state
            policy_q_values = self.policy_dqn(state.unsqueeze(0))
            current_q = policy_q_values[0, action]
            current_q_list.append(current_q)

            # Get Q-values from target for next state and compute target Q-value
            with torch.no_grad():
                # Get Q-values from target for next state
                target_q_values = self.target_dqn(next_state.unsqueeze(0))

                # Calculate target Q-value
                if done:
                    target = reward
                else:
                    # Bellman Equation!
                    target = reward + (self.discount_factor * torch.max(target_q_values).item())

                target_q_list.append(target)

        
        # Compute loss for batch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
