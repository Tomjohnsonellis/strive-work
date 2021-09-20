from paths import generate_possible_paths

# Test 1
board_one = [4,3]
snake_one = [[2,2], [3,2], [3,1], [3,0], [2,0], [1,0], [0,0]]
depth_one = 3

valid, _ = generate_possible_paths(depth_one, snake_one, board_one)
print(len(valid))

# Test 2
board_two = [2,3]
snake_two = [[0,2], [0,1], [0,0], [1,0], [1,1], [1,2]]
depth_two = 10

# This test crashes my pc, probably due to the millions of combinations
# valid, _ = generate_possible_paths(depth_two, snake_two, board_two)
# print(len(valid))

# Test 3
board_three = [10,10]
snake_three = [[5,5], [5,4], [4,4], [4,5]]
depth_three = 4
valid, _ = generate_possible_paths(depth_two, snake_two, board_two)
print(len(valid))