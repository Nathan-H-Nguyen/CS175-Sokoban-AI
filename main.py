from Board import *
from AI import SokobanSolver
from time import time

#/Users/kaydeereyes/PycharmProjects/CS175-Sokoban-AI/Sokoban-benchmarks/Sokoban/sokoban-01.txt
if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = "Sokoban-benchmarks/Sokoban/" + input("Enter the input path filename (e.g., 'input.txt'): ")
    else:
        filename = sys.argv[1]
    board = Board(filename)

    version = input("Enter version type: ")


    valid = {'L', 'R', 'U', 'D'}
    version_valid = {'M', 'BFS'}

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
    if version == 'BFS':
        start = time()
        solver = SokobanSolver(board)
        path = solver.solve_BFS()
        elapsed_time = time()-start

        if not path:
            print("No solution found.")
        else:
            print("\nSolution found!")
            print(f"Time to solve: {elapsed_time}s")
            print("Moves:", ''.join(path))
            print('\n')
            board.print()
            print('\n')
            for move in path:
                print(f"Move: {move}")
                board.move(move)
                board.print()
                print('\n')

            print("Game Won!")