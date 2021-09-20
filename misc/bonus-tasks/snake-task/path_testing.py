from generators import generate_snake
from movement import move_snake
import itertools # Part of base python!

# Parameters
length_of_snake = 5
board = [4,5]

snake_to_test = generate_snake(length_of_snake, board)
print(snake_to_test)
directions = ["U","D","L","R"]
depth = 4

possible_paths = itertools.permutations(directions, depth)
valid_paths = []
invalid_paths = []
for path in possible_paths:
    move_no = 0
    print(f"SNAKETOTEST: {snake_to_test}")

    theoretical_snake = snake_to_test
    print(f"At the start, snake is: {theoretical_snake}")
    print(f"Attempting path: {path}")
    for direction in path:
        print(f"Moving: {direction}")
        theoretical_snake, valid = move_snake(direction, theoretical_snake, board)
        print(f"Move validity: {valid}")
        if valid == False:
            print("INVALID PATH")
            invalid_paths.append([path])
            break
        print(f"Current snake: {theoretical_snake}")
        move_no += 1
        if move_no == depth:
            print("Valid move!")
            valid_paths.append([path])

    print("-"*50)
    
print(f"VALIDS: {len(valid_paths)}")
print(valid_paths)
print(f"INVALIDS: {len(invalid_paths)}")
print(invalid_paths)


# def generate_paths(depth, snake, board=board):
#     possible_paths = itertools.combinations_with_replacement(directions, depth)
#     valid_paths = []
#     invalid_paths = []
#     original_snake = snake
#     for single_path in possible_paths:
#         path_is_valid = True
#         theoretical_snake = original_snake
#         print("-"*50)
#         print(f"Snake: {theoretical_snake}")
#         print(f"Path: {single_path}")
#         print("-"*50)

#         for direction in single_path:
#             print(f"Moving: {direction}")
#             theoretical_snake, valid = move_snake(direction, theoretical_snake)
#             print(valid)
            
#             print(f"Snake after move: {theoretical_snake}")
#             if not valid:
#                 path_is_valid = False

#         if path_is_valid:
#             valid_paths.append(single_path)
#         else:
#             invalid_paths.append(single_path)
#         print("NEXT PATH")

#     # return
#     return valid_paths, invalid_paths

# # generate_paths(5, snake)
# good, bad = generate_paths(depth, snake)
# print("~"*50)
# print(snake)
# print("~"*50)

# print(f"Good paths:\n{good}")
# print(f"Bad paths:\n{bad}")

    