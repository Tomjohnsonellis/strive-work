# from generators import generate_snake
snake_array = list[int,int,]
# Defining variables to compile sucessfully
# board = [4,5]
# snake = generate_snake()


def move_snake(direction:str, snake_to_move, board_array) -> snake_array:
    if direction == "U":
        moved_snake, valid = move_up(snake_to_move, board_array)
    if direction == "D":
        moved_snake, valid = move_down(snake_to_move, board_array)
    if direction == "L":
        moved_snake, valid = move_left(snake_to_move, board_array)
    if direction == "R":
        moved_snake, valid = move_right(snake_to_move, board_array)

    return moved_snake, valid

def move_up(snake_to_move, board_array):
    
    potential_new_head = [snake_to_move[0][0] - 1, snake_to_move[0][1] ]
    # Is it possible?
    if potential_new_head[0] == -1:
        # print("Invalid move: UP - Wall")
        return snake_to_move, False
    # Is it legal?
    if potential_new_head in snake_to_move:
        # print("Invalid move: UP - Snake")
        return snake_to_move, False

    # snake.insert(0, potential_new_head)
    # # print(snake)
    # new_snake = snake[0:-1]
    # # print(new_snake)

    dummy_snake = [potential_new_head]
    for body_part in snake_to_move:
        dummy_snake.append(body_part)
    new_snake = dummy_snake[0:-1]


    return new_snake, True

def move_down(snake_to_move, board_array):
    potential_new_head = [snake_to_move[0][0] + 1, snake_to_move[0][1]]
    # Is it possible?
    if potential_new_head[0] > board_array[0]:
        # print("Invalid move: DOWN - Wall")
        return snake_to_move, False
    # Is it legal?
    if potential_new_head in snake_to_move:
        # print("Invalid move: DOWN - Snake")
        return snake_to_move, False

    # snake.insert(0, potential_new_head)
    # # print(snake)
    # new_snake = snake[0:-1]
    # # print(new_snake)

    dummy_snake = [potential_new_head]
    for body_part in snake_to_move:
        dummy_snake.append(body_part)
    new_snake = dummy_snake[0:-1]

    return new_snake, True


def move_left(snake_to_move, board_array):
    potential_new_head = [snake_to_move[0][0], snake_to_move[0][1] - 1]
    # Is it possible?
    if potential_new_head[1] == -1:
        # print("Invalid move: LEFT - Wall")
        return snake_to_move, False
    # Is it legal?
    if potential_new_head in snake_to_move:
        # print("Invalid move: LEFT - Snake")
        return snake_to_move, False
    
    # snake.insert(0, potential_new_head)
    # new_snake = snake[0:-1]
    
    dummy_snake = [potential_new_head]
    for body_part in snake_to_move:
        dummy_snake.append(body_part)
    new_snake = dummy_snake[0:-1]

    return new_snake, True

def move_right(snake_to_move, board_array):
    potential_new_head = [snake_to_move[0][0], snake_to_move[0][1] + 1]
     # Is it possible?
    if potential_new_head[1] > board_array[1]:
        # print("Invalid move: RIGHT - Wall")
        return snake_to_move, False
    # Is it legal?
    if potential_new_head in snake_to_move:
        # print("Invalid move: RIGHT - Snake")
        return snake_to_move, False
    

    # snake.insert(0, potential_new_head)
    # new_snake = snake[0:-1]

    dummy_snake = [potential_new_head]
    for body_part in snake_to_move:
        dummy_snake.append(body_part)
    new_snake = dummy_snake[0:-1]

    return new_snake, True

