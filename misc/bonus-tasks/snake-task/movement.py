from generators import generate_snake
snake_array = list[int,int,]
# Defining variables to compile sucessfully
board = [4,5]
snake = generate_snake()

def move_snake(direction:str, snake=snake, board=board) -> snake_array:
    if direction == "U":
        snake, valid = move_up(snake, board)
    if direction == "D":
        snake, valid = move_down(snake, board)
    if direction == "L":
        snake, valid = move_left(snake, board)
    if direction == "R":
        snake, valid = move_right(snake, board)

    return snake, valid

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
    if potential_new_head[1] == -1:
        # print("Invalid move: LEFT - Wall")
        return snake, False
    # Is it legal?
    if potential_new_head in snake:
        # print("Invalid move: LEFT - Snake")
        return snake, False
    
    snake.insert(0, potential_new_head)
    new_snake = snake[0:-1]
    return new_snake, True

def move_right(snake=snake, board=board):
    potential_new_head = [snake[0][0], snake[0][1] + 1]
     # Is it possible?
    if potential_new_head[1] > board[1]:
        # print("Invalid move: RIGHT - Wall")
        return snake, False
    # Is it legal?
    if potential_new_head in snake:
        # print("Invalid move: RIGHT - Snake")
        return snake, False
    

    snake.insert(0, potential_new_head)
    new_snake = snake[0:-1]
    return new_snake, True

