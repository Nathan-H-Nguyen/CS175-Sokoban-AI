import os
import random
import time
import torch
from Board import Board
from DQNEnv import DQNEnv
from DQN import Agent

ALPHA = 0.001
GAMMA = 0.99
BATCH_SIZE = 64
SYNC_RATE = 3000

if __name__ == '__main__':
    # Get all level paths
    easy_path = os.path.join(os.getcwd(), 'Train/Sokoban/Easy')
    medium_path = os.path.join(os.getcwd(), 'Train/Sokoban/Medium')
    hard_path = os.path.join(os.getcwd(), 'Train/Sokoban/Hard')

    # Get all levels
    easy_levels = list(os.listdir(easy_path))
    medium_levels = list(os.listdir(medium_path))
    hard_levels = list(os.listdir(hard_path))

    # Get Training and Test Levels
    #random.shuffle(easy_levels) # Randomize
    easy_len = len(easy_levels)
    easy_split = int(easy_len*0.80)
    easy_train = easy_levels[:easy_split] # 80% Train
    easy_test = easy_levels[easy_split:] # 20% Test

    #random.shuffle(medium_levels) # Randomize
    medium_len = len(medium_levels)
    medium_split = int(medium_len*0.80)
    medium_train = medium_levels[:medium_split] # 80% Train
    medium_test = medium_levels[medium_split:] # 20% Test

    #random.shuffle(hard_levels) # Randomize
    hard_len = len(hard_levels)
    hard_split = int(hard_len*0.80)
    hard_train = hard_levels[:hard_split] # 80% Train
    hard_test = hard_levels[hard_split:] # 20% Test

    # Find largest in_states (Hard has the largest board so only check those levels)
    max_size = 0
    for level in hard_levels:
        board = Board(hard_path + '/' + level)
        if board.rows*board.cols > max_size:
            max_size = board.rows*board.cols
    
    # Initialize Agent
    agent = Agent(None, learning_rate=ALPHA, discount_factor=GAMMA, epsilon=1.00, batch_size=BATCH_SIZE, sync_rate=SYNC_RATE, 
                    in_states=max_size, h1_nodes=128, h2_nodes=128, maxlen=30_000)

    """==================== TRAINING ===================="""
    with open('Training_Results.txt', 'a') as f:
        total_start = time.time()
        # EASY TRAINING
        f.write('====================EASY TRAINING====================\n')
        easy_start = time.time() # Track how long to complete easy training
        for level in easy_train:
            level_start = time.time()
            board = Board(easy_path + '/' + level)
            env = DQNEnv(board, max_size, 200 + 50*board.num_boxes)
            agent.env = env
            agent.train(500)

            f.write(f'{level}:\n')
            f.write(f'\tTime to train: {time.time()-level_start}s\n')
        
        f.write(f'Total time to train: {time.time()-easy_start}s\n\n')

        # EASY AND MEDIUM TRAINING
        f.write('====================EASY AND MEDIUM TRAINING====================\n')
        easy_and_medium_start = time.time() # Track how long to complete easy and medium training
        agent.epsilon = 0.20
        agent.epsilon_start = 0.20
        agent.epsilon_min = 0.05
        agent.epsilon_half_life = 750_000
        agent.epsilon_step_counter = 0

        #Combine easy and medium levels together then shuffle
        easy_combine = [{'difficulty': 'easy', 'level': level} for level in easy_train]
        medium_combine = [{'difficulty': 'medium', 'level': level} for level in medium_train]
        easy_and_medium = easy_combine + medium_combine
        random.shuffle(easy_and_medium)

        for item in easy_and_medium:
            difficulty = item['difficulty']
            level = item['level']
            if difficulty == 'easy':
                level_start = time.time()
                board = Board(easy_path + '/' + level)
                env = DQNEnv(board, max_size, 200 + 50*board.num_boxes)
                agent.env = env
                agent.train(500)

                f.write(f'{level}:\n')
                f.write(f'\tTime to train: {time.time()-level_start}s\n')
            else:
                level_start = time.time()
                board = Board(medium_path + '/' + level)
                env = DQNEnv(board, max_size, 300 + 75*board.num_boxes)
                agent.env = env
                agent.train(500)

                f.write(f'{level}:\n')
                f.write(f'\tTime to train: {time.time()-level_start}s\n')
            
        f.write(f'Total time to train: {time.time()-easy_and_medium_start}s\n\n')
        # MEDIUM AND HARD TRAINING
        medium_and_hard_start = time.time() # Track how long to complete medium and hard training
        agent.epsilon = 0.30
        agent.epsilon_start = 0.30
        agent.epsilon_min = 0.05
        agent.epsilon_half_life = 1_000_000
        agent.epsilon_step_counter = 0

        # Combine medium and hard then shuffle
        medium_combine = [{'difficulty': 'medium', 'level': level} for level in medium_train]
        hard_combine = [{'difficulty': 'hard', 'level': level} for level in hard_train]
        medium_and_hard = medium_combine + hard_combine
        random.shuffle(medium_and_hard)

        for item in medium_and_hard:
            difficulty = item['difficulty']
            level = item['level']

            if difficulty == 'medium':
                level_start = time.time()
                board = Board(medium_path + '/' + level)
                env = DQNEnv(board, max_size, 300 + 75*board.num_boxes)
                agent.env = env
                agent.train(500)

                f.write(f'{level}:\n')
                f.write(f'\tTime to train: {time.time()-level_start}s\n')
            else:
                level_start = time.time()
                board = Board(hard_path + '/' + level)
                env = DQNEnv(board, max_size, 500 + 100*board.num_boxes)
                agent.env = env
                agent.train(500)

                f.write(f'{level}:\n')
                f.write(f'\tTime to train: {time.time()-level_start}s\n')

        f.write(f'Total time to train: {time.time()-medium_and_hard_start}s\n\n')
        f.write('============================================================\n')
        f.write(f'Total training time: {time.time()-total_start}s')
    
    """==================== TESTING ===================="""
    # TESTING DONE HOPEFULLY WE GOT GOOD RESULTS LETS TEST ON THE OTHER 20%
    agent.policy_dqn.load_state_dict(torch.load('dqn.pt'))
    agent.target_dqn.load_state_dict(agent.policy_dqn.state_dict())
    with open('Test_Results.txt', 'a') as f:
        f.write('====================EASY TEST RESULTS====================\n')
        print('====================EASY TEST RESULTS====================')
        for level in easy_test:
            path = easy_path + '/' + level
            board = Board(path)
            win, total_steps, accumulated_reward = agent.test(board, max_size, 200 + 50*board.num_boxes)

            f.write(f'{level}:\n')
            f.write(f'\tWin: {win}\n')
            f.write(f'\tTotal Steps: {total_steps}\n')
            f.write(f'\tAccumulated Reward: {accumulated_reward}\n')
            print(f'{level}:')
            print(f'\tWin: {win}')
            print(f'\tTotal Steps: {total_steps}')
            print(f'\tAccumulated Reward: {accumulated_reward}')

        
        f.write('===================MEDIUM TEST RESULTS===================\n')
        print('===================MEDIUM TEST RESULTS===================')
        for level in medium_test:
            path = medium_path + '/' + level
            board = Board(path)
            win, total_steps, accumulated_reward = agent.test(board, max_size, 300 + 75*board.num_boxes)
            f.write(f'{level}:\n')
            f.write(f'\tWin: {win}\n')
            f.write(f'\tTotal Steps: {total_steps}\n')
            f.write(f'\tAccumulated Reward: {accumulated_reward}\n')

            print(f'{level}:')
            print(f'\tWin: {win}')
            print(f'\tTotal Steps: {total_steps}')
            print(f'\tAccumulated Reward: {accumulated_reward}')
        
        f.write('====================HARD TEST RESULTS====================\n')
        print('====================HARD TEST RESULTS====================')
        for level in hard_test:
            path = hard_path + '/' + level
            board = Board(path)
            win, total_steps, accumulated_reward, actions = agent.test(board, max_size, 500 + 100*board.num_boxes)
            f.write(f'{level}:\n')
            f.write(f'\tWin: {win}\n')
            f.write(f'\tTotal Steps: {total_steps}\n')
            f.write(f'\tAccumulated Reward: {accumulated_reward}\n')
            f.write(f'\tActions: {actions}')

            print(f'{level}:')
            print(f'\tWin: {win}')
            print(f'\tTotal Steps: {total_steps}')
            print(f'\tAccumulated Reward: {accumulated_reward}')
            print(f'\tACtions: {actions}')