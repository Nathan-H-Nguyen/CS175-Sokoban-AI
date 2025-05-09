from Board import *
from AI import SokobanSolver

#/Users/kaydeereyes/PycharmProjects/CS175-Sokoban-AI/Sokoban-benchmarks/Sokoban/sokoban-01.txt
if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = input("Enter the input path filename (e.g., 'input.txt'): ")
    else:
        filename = sys.argv[1]
    board = Board(filename)

    version = input("Enter version type: ")


    valid = {'L', 'R', 'U', 'D'}
    version_valid = {'M', 'BFS'}
    board.print()

    if version == 'M':
        while not board.is_win():
            direction = input("Enter a direction to move (L, R, U, D): ").strip().upper()
            if direction not in valid:
                print(f"Invalid direction '{direction}'.")
                continue
            board.move(direction)
            board.print()

        print("Game Won!")
    if version == 'BFS':
        while not board.is_win():
            solver = SokobanSolver(board)
            path = solver.solve_BFS()

            if not path:
                print("No solution found.")
            else:
                print("Solution found!")
                print("Moves:", ''.join(path))
                for move in path:
                    board.move(move)
                    board.print()

                print("Game Won!")