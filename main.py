from Board import *
from BFS import BFS
from Astar import AStar
from time import time
from IDA import IDA

if __name__ == '__main__':
    test = True
    test_version = "IDA"
    test_filename = "Sokoban-benchmarks/Sokoban/sokoban0000.txt"

    if len(sys.argv) < 2:
        filename = test_filename if test else "Sokoban-benchmarks/Sokoban/" + input("Enter the input path filename (e.g., 'input.txt'): ")
    else:
        filename = sys.argv[1]
    board = Board(filename)

    version = test_version if test else input("Enter version type (M, BFS, A*): ").upper()

    valid = {'L', 'R', 'U', 'D'}
    version_valid = {'M', 'BFS', 'A*', 'A*2'}


    if version == 'M':
        board.print()
        while not board.is_win():
            direction = input("Enter a direction to move (L, R, U, D): ").strip().upper()
            if direction not in valid:
                print(f"Invalid direction '{direction}'.")
                continue
            board.move(direction)
            board.print()

        print("Game Won!")


    elif version == 'BFS':
        start = time()
        solver = BFS(board)
        path = solver.solve()
        elapsed_time = time()-start

        with open('Output/BFS.txt', 'a') as f:
            board.write_output(f, filename, elapsed_time, path)


    elif version == 'A*':
        start = time()
        solver = AStar(board)
        path = solver.solve()
        elapsed_time = time()-start

        with open('Output/AStar.txt', 'a') as f:
            board.write_output(f, filename, elapsed_time, path)

    elif version == 'IDA':
        start = time()
        solver = IDA(board)
        path = solver.solve()
        elapsed_time = time() - start

        with open('Output/IDA.txt', 'a') as f:
            board.write_output(f, filename, elapsed_time, path)



