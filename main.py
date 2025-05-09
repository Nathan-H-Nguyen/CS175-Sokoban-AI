import sys
from Board import Board

if __name__ == '__main__':
    board = Board(sys.argv[1])
    valid = {'L', 'R', 'U', 'D'}
    lose_flag = 0

    board.print()
    while not board.is_win():

        direction = input("Enter a direction to move (L, R, U, D): ").strip().upper()
        if direction not in valid:
            print(f"Invalid direction '{direction}'.")
            continue
        board.move(direction)
        board.print()

        if board.box_corner_trap():
            lose_flag = 1
            break
    
    if not lose_flag:
        print("Game Won!")
    else:
        print("Game Lost due to deadlock.")