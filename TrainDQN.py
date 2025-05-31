import os
from Board import Board
from DQNEnv import DQNEnv
from DQN import Agent

EPISODES = 200

if __name__ == '__main__':
    # Path to train levels
    training_path = os.path.join(os.getcwd(), 'Train/Sokoban')

    # Get levels
    levels = sorted(f for f in os.listdir(training_path))

    # Get max size of board in training levels
    max_size = 0
    for level in levels:
        board = Board(training_path + '/' + level)
        if board.rows*board.cols > max_size:
            max_size = board.rows*board.cols

    # Setup Agent for training
    agent = Agent(None, learning_rate=0.0001, discount_factor=0.99, epsilon=1.00, batch_size=64, sync_rate=3000, 
                    in_states=max_size, h1_nodes=128, h2_nodes=128, maxlen=30_000)

    for level in levels:
        print(f'TRAINING: {level}')
        board = Board(training_path + '/' + level)
        agent.env = DQNEnv(board, max_size, board.num_boxes*1000)
        agent.train(EPISODES)