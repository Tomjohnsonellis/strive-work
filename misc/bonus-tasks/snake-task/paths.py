from generators import generate_snake
from movement import move_snake
import itertools # Part of base python!

# Parameters
length_of_snake = 5
board = [4,5]

snake_to_test = generate_snake(length_of_snake, board)
directions = ["U","D","L","R"]
depth = 99

def generate_possible_paths(depth, snake, board):
    
    board = fix_board(board)
    directions = ["U","D","L","R"] * depth # I was having difficulty generating ALL possible paths, this is overkill
    possible_paths = itertools.combinations_with_replacement(directions, depth)

    # print("x-"*50)
    # for path in possible_paths:
    #     print(path)
    # print("x-"*50)
    valid_paths = []
    invalid_paths = 0
    original_snake = snake
    for path in possible_paths:
        move_no = 0
        # Reset to the original snake each time
        testing_snake = original_snake

        # # TESTING
        # print("@"*50)
        # print(f"testing_snake: {testing_snake}")
        # print(f"path: {path}")

        for direction in path:
            # print(f"Moving: {direction}")
            testing_snake, valid = move_snake(direction, testing_snake, board)
            # print(f"VALID: {valid}")
            # print(f"current snake: {testing_snake}")

            # At any point, an invalid move means that the path is not valid
            
            if valid == False:
                # Removed due to memory issues
                # invalid_paths.append((path))
                break
            
            # If we manage to do <depth> moves sucessfully, it's a valid path
            move_no += 1
            if move_no == depth:
                valid_paths.append((path))
                # print("~VALID~" *50)

    # Remove any duplicates
    valid_set = set(valid_paths)
    # Removed due to memory issues
    # invalid_set = set(invalid_paths)
    invalid_set = 0

    # for val in valid_paths:
    #     valid_set.add(val)
    # for inv in invalid_paths:
    #     invalid_set.add(inv)

    # Returns two lists
    return valid_set, invalid_set

def fix_board(board): 
    # I messed up the board logic somewhere, but this fixes it.
    fixed_board = board
    fixed_board[0] = board[0]-1
    fixed_board[1] = board[1]-1
    return fixed_board

if __name__ == "__main__":
    val, inv = generate_possible_paths(3, snake_to_test, board)
    print(len(val))
    print(snake_to_test)
    print(type(val))
    print(set(val))