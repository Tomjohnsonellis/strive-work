from generators import generate_snake
from movement import move_snake
import itertools # Part of base python!

# Parameters
length_of_snake = 5
board = [4,5]

snake_to_test = generate_snake(length_of_snake, board)
directions = ["U","D","L","R"]
depth = 5

def generate_possible_paths(depth, snake, board):
    directions = ["U","D","L","R"]
    possible_paths = itertools.permutations(directions, depth)
    valid_paths = []
    invalid_paths = []
    original_snake = snake
    for path in possible_paths:
        move_no = 0
        # Reset to the original snake each time
        testing_snake = original_snake

        for direction in path:
            testing_snake, valid = move_snake(direction, testing_snake, board)

            # At any point, an invalid move means that the path is not valid
            if valid == False:
                invalid_paths.append([path])
                break
            
            # If we manage to do <depth> moves sucessfully, it's a valid path
            move_no += 1
            if move_no == depth:
                valid_paths.append([path])

    # Returns two lists
    return valid_paths, invalid_paths

val, inv = generate_possible_paths(3, snake_to_test, board)
print(len(val), len(inv))
print(snake_to_test)
print(val)