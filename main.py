from Board import *
from BFS import BFS
from AStar import AStar
from time import time

if __name__ == '__main__':
    test = True
    test_version = "A*"
    test_filename = "Sokoban-benchmarks/Sokoban/sokoban01.txt"

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
            f.write(f'Input File: {filename}\n\n')
            if not path:
                f.write("No solution found.\n\n")
                f.write("#########################################################################################################################################################################")
                f.write('\n\n')

                print("No solution found.")
            else:
                f.write(f"Time to solve: {elapsed_time}s\n")
                print(f"Time to solve: {elapsed_time}s")

                f.write(f"Number of moves: {len(path)}\n")
                print(f"Number of moves: {len(path)}")

                moves = "Moves:" + ''.join(path)
                f.write(moves + '\n')
                print(moves)

                f.write('\n\n')
                print('\n')

                board.print_to_file(f)
                board.print()

                f.write('\n\n')
                print('\n')
                for move in path:
                    f.write(f"Move: {move}\n")
                    print(f"Move: {move}")

                    board.move(move)

                    board.print_to_file(f)
                    board.print()

                    f.write('\n\n')
                    print('\n')

                f.write("Game Won!\n\n")
                f.write("#########################################################################################################################################################################")
                f.write('\n\n')

                print("Game Won!")
    elif version == 'A*':
        start = time()
        solver = AStar(board)
        path = solver.solve()
        elapsed_time = time()-start

        with open('Output/AStar.txt', 'a') as f:
            f.write(f'Input File: {filename}\n\n')
            if not path:
                f.write("No solution found.\n\n")
                f.write("#########################################################################################################################################################################")
                f.write('\n\n')
                print("No solution found.")
            else:
                f.write(f"Time to solve: {elapsed_time}s\n")
                print(f"Time to solve: {elapsed_time}s")

                f.write(f"Number of moves: {len(path)}\n")
                print(f"Number of moves: {len(path)}")

                moves = "Moves:" + ''.join(path)
                f.write(moves + '\n')
                print(moves)

                f.write('\n\n')
                print('\n')

                board.print_to_file(f)
                board.print()

                f.write('\n\n')
                print('\n')
                for move in path:
                    f.write(f"Move: {move}\n")
                    print(f"Move: {move}")

                    board.move(move)

                    board.print_to_file(f)
                    board.print()

                    f.write('\n\n')
                    print('\n')

                f.write("Game Won!\n\n")
                f.write("#########################################################################################################################################################################")
                f.write('\n\n')

                print("Game Won!")
