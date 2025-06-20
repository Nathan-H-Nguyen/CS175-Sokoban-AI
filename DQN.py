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
from typing import List, Dict
import math

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

        # Epsilon stuff
        self.epsilon = epsilon # epsilon-greedy
        # Use these for training on Easy, change when training on other levels
        self.epsilon_start = epsilon
        self.epsilon_min = 0.05
        self.epsilon_half_life = 250_000
        self.epsilon_step_counter = 0

        self.batch_size = batch_size
        self.sync_rate = sync_rate
        self.sync_step_counter = 0

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

        self.train_count = 1
    
    def select_action(self, state: torch.Tensor) -> int:
        """
        Selects an action from the given state, random or best move

        Args:
            state (torch.Tensor): 1D Tensor representing state of board/environment

        Returns:
            int: Action to take (0123 maps to LRUD respectively)
        """
        # Increment step
        self.sync_step_counter += 1
        self.epsilon_step_counter += 1

        # Epsilon-greedy policy to randomly decide on choosing random move or "best" move
        if random.random() < self.epsilon:
            return random.randint(0, 3) # 0123 => LRUD
        else:
            q_values = self.policy_dqn(state.unsqueeze(0)) # Get q values from output layer
            return torch.argmax(q_values).item() # Return best move
        
    # Since were training across multiple levels we want to decay epsilon according to that, we do an exponential decay thing
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
            steps_to_finish = 0

            # While puzzle hasnt been solved and havent exceeded step limit
            while (not done):
                # Select Action for current state
                action = self.select_action(state)

                # Make Action, get results
                new_state, reward, done, info = self.env.step(action)
                accumulated_reward += reward
                steps_to_finish += 1

                # Append Transition to Memory
                transition = Transition(state, action, reward, new_state, done)
                self.memory.append(transition)

                # Update State
                state = new_state

                # Grab Sample Batch from Memory and Optimize
                if len(self.memory) >= self.batch_size:
                    batch = self.memory.sample(self.batch_size)
                    self.optimize(batch)

                # Sync policy_dqn and target_dqn
                if self.sync_step_counter >= self.sync_rate:
                    self.sync_step_counter = 0
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

                # Exponential decay
                self.epsilon = max(self.epsilon_min + (self.epsilon_start - self.epsilon_min) * math.exp(-self.epsilon_step_counter/self.epsilon_half_life), self.epsilon_min)
            
            print(f'{os.path.basename(self.env.board.file)} - Episode {i} finished after: {steps_to_finish} steps')
            
            # Lil visual debug thing
            if steps_to_finish == self.env.step_limit:
                self.env.board.print()
            
            # Track changes per episode
            rewards_per_episode[i] = accumulated_reward
            epsilon_history[i] = self.epsilon
            
        
        # Save Policy
        torch.save(self.policy_dqn.state_dict(), 'dqn.pt')

        self._plot_results(rewards_per_episode, epsilon_history)

    def optimize(self, batch: List[Transition]) -> None:
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for transition in batch:
            states.append(transition.state)
            actions.append(transition.action)
            rewards.append(transition.reward)
            next_states.append(transition.next_state)
            dones.append(transition.done)
        
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack(next_states)
        dones = torch.tensor((dones), dtype=torch.bool).unsqueeze(1)

        policy_q_values = self.policy_dqn(states) # Get all Q-values from policy
        current_q = policy_q_values.gather(1, actions) # Get action specific Q-value

        # Double Q Learn
        with torch.no_grad():
            # # Get action from policy network
            # policy_q_values_next = self.policy_dqn(next_states)
            # policy_actions = torch.argmax(policy_q_values_next, dim=1, keepdim=True)

            # # Use target nework
            # target_q_values   = self.target_dqn(next_states) # Get all Q-values from target
            # q_values = target_q_values.gather(1, policy_actions)

            # # GENIUS if the transition is done (True) for whatever reason, only use reward, else (False) do entire equation
            # targets  = rewards + (~dones) * self.discount_factor * q_values # Get highest Q-value

            # Single Q Learning
            target_q_values = self.target_dqn(next_states)
            targets = rewards + (~dones) * self.discount_factor * target_q_values.max(1, keepdim=True)[0]

        # Compute loss for batch
        loss = self.loss_fn(current_q, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_dqn.parameters(), 5.0) # Prevents outliers messing up training
        self.optimizer.step()

    def test(self, board: Board, max_size: int, step_limit: int) -> Dict[str, object]:
        """
        Runs a greedy episode on the given level

        Args:
            board (Board): Board object of level
            max_size (int): Max size of the environment
            step_limit (int): Max number of steps allowed for the given level
        
        Returns:
            Dict[str, object]: Dict containing info regarding the test
        """
        env = DQNEnv(board, max_size, step_limit)
        self.env = env

        prev_epsilon = self.epsilon
        self.epsilon = 0.0 # Force greedy action

        state = env.reset()
        accumulated_reward = 0
        total_steps = 0
        done = False
        actions = []

        while not done and total_steps < step_limit:
            action = self.select_action(state) # Get move
            actions.append(action)
            next_state, reward, done, info = env.step(action) # Make Move

            # Update metrics
            accumulated_reward += reward
            total_steps += 1
            state = next_state
        
        self.epsilon = prev_epsilon

        return {'win': info['win'], 'total_steps': total_steps, 'accumulated_reward': accumulated_reward, 'actions': actions}

    def _plot_results(self, rewards_per_episode, epsilon_history):
        episodes = np.arange(len(rewards_per_episode))
        
        # 20 episode window average
        window = 20       
        weights = np.ones(window) / window
        smoothed = np.convolve(rewards_per_episode, weights, mode="same")
        
        plt.figure(figsize=(24, 8), dpi=100)

        # Accumulated Reward subplot
        ax1 = plt.subplot(1, 2, 1)

        # Plot raw data faintly (Every 5th episode)
        step = 5
        ax1.plot(episodes[::step], rewards_per_episode[::step], color="blue", alpha=0.25, label=f"Raw (every {step}th)")

        # Plot window average
        ax1.plot(episodes, smoothed, color="orange", linewidth=2.5, label=f"{window}-episode moving avg")

        ax1.set_title(f"{os.path.basename(self.env.board.file)}: Accumulated Reward per Episode", fontsize=20)
        ax1.set_xlabel("Episode", fontsize=16)
        ax1.set_ylabel("Accumulated Reward", fontsize=16)
        ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax1.legend(fontsize=12)

        # Tick every 50 episodes on x axis
        ax1.set_xticks(np.arange(0, len(episodes)+1, 50))
        ax1.set_xticklabels([str(x) for x in np.arange(0, len(episodes)+1, 50)], fontsize=12)
        ax1.tick_params(axis="y", labelsize=12)

        # Epsilon Decay subplot
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(episodes, epsilon_history, color="green", linewidth=2)
        ax2.set_title(f"{os.path.basename(self.env.board.file)}: Epsilon Decay per Episode", fontsize=20)
        ax2.set_xlabel("Episode", fontsize=16)
        ax2.set_ylabel("Epsilon", fontsize=16)
        ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Tick every 50 episode on x axis
        ax2.set_xticks(np.arange(0, len(episodes)+1, 50))
        ax2.set_xticklabels([str(x) for x in np.arange(0, len(episodes)+1, 50)], fontsize=12)
        ax2.tick_params(axis="y", labelsize=12)

        plt.tight_layout()
        plt.savefig(f"DQN_Training_Results_PNG/dqn_training_results_{self.train_count}_{os.path.basename(self.env.board.file)}.png")
        plt.close()
        
        self.train_count += 1
