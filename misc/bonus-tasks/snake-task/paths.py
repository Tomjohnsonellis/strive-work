from generators import generate_snake
from movement import move_snake

# Parameters
length_of_snake = 5
board = [4,5]

snake = generate_snake(length_of_snake, board)

snake, valid = move_snake("R")
print(snake, valid)
snake, valid = move_snake("R")
print(snake, valid)
snake, valid = move_snake("R")
print(snake, valid)
snake, valid = move_snake("R")
print(snake, valid)
    