from generators import generate_snake
snake_array = list[int,int,]
# Parameters
length_of_snake = 5
board = [4,5]

snake = generate_snake(length_of_snake, board)

print(snake)

# snake[0][1] is y
# snake[0][0] is x

def move_snake(direction:str, snake=snake, board=board) -> snake_array:
    if direction == "U":
        pass
    if direction == "D":
        pass
    if direction =="L":
        pass
    if direction == "R":
        pass

    return snake

def move_up(snake=snake, board=board):
    potential_new_head = [snake[0][0] - 1, snake[0][1] ]
    # Is it possible?
    if potential_new_head[0] == -1:
        # print("Invalid move: UP - Wall")
        return snake, False
    # Is it legal?
    if potential_new_head in snake:
        # print("Invalid move: UP - Snake")
        return snake, False

    snake.insert(0, potential_new_head)
    # print(snake)
    new_snake = snake[0:-1]
    # print(new_snake)

    return new_snake, True

def move_down(snake=snake, board=board):
    potential_new_head = [snake[0][0] + 1, snake[0][1]]
    # Is it possible?
    if potential_new_head[0] > board[0]:
        # print("Invalid move: DOWN - Wall")
        return snake, False
    # Is it legal?
    if potential_new_head in snake:
        # print("Invalid move: DOWN - Snake")
        return snake, False

    snake.insert(0, potential_new_head)
    # print(snake)
    new_snake = snake[0:-1]
    # print(new_snake)

    return new_snake, True


def move_left(snake=snake, board=board):
    potential_new_head = [snake[0][0], snake[0][1] - 1]
    # Is it possible?
    





def move_right(snake=snake, board=board):
    potential_new_head = [snake[0][0], snake[0][1] + 1]



snake, valid = move_down()
print(snake)
print(valid)