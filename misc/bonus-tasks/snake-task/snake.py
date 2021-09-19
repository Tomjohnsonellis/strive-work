"""
Snake pathfinder task

Given:
A board - b,
A snake - s,
A depth - d,
How many valid paths of length d are available?

Constraints:
board max size is [10,10]
board[0] represents rows
board[1] represents columns
No additional info in board

3 <= snake length <= 7
snake[x] contains the coordinates of each section on snake
snake[0] is the head
snake[i] and snake [i+1] are adjacent (horizontally or vertically)
the snake's initial positions are valid (no self intersection, entirely on the board)
snake[i][j] < board[j] (Each snake part is actully in bounds)


No frameworks (numpy, pandas etc), base language capabilities only. (import random is fine)

"""
# No third party imports!
import random

def validate_board(board_array:list[int]) -> bool:
    valid = True
    for i in board_array:
        if not i <= 10:
            valid = False

    return valid

# Co-ordsinates are: Rows, Columns.
# So 3x5 looks like...
# 0 1 2 3 4
# 1 
# 2
board = [4,5]
print(f"Board valid?: {validate_board(board)}")

def generate_snake(snake_length:int=10, board_array:list[int,int]=board) -> list[list[int,int]]:
    y = random.randint(0, board_array[0])
    x = random.randint(0, board_array[1])
    snake_head = [y,x]

    # TESTING
    snake_head = [3,4]

    snake = [snake_head]
    # print(snake_head)
    create_snake_body(snake, snake_length)


    # Create X blocks
    # Use the length and just create all the blocks at once



    return snake

def create_snake_body(snake:list[list[int,int]], snake_length:int, board_array:list[int,int]=board) -> list[list[int,int]]:
    pieces_to_generate = snake_length - 1
    
    # The idea here is to build the snake in a way that avoids the snake extending itself into a corner
    mid_y = board_array[0]/2
    mid_x = board_array[1]/2
    


    # We'll split the board into 4 sections, and the section that the snake head is in determines how it will be extended
    pos_top = False
    pos_left = False
    if snake[0][0] <= mid_y:
        pos_top = True
    else:
        pos_top = False
    
    if snake[0][1] <= mid_x:
        pos_left = True
    else:
        pos_left = False

    # TESTING
    print(f"MID POINT: {mid_y, mid_x}")
    print(f"SNAKE HEAD: {snake[0]}")
    print(f"TOP: {pos_top}")
    print(f"LEFT: {pos_left}")

    generation_point = snake[0]
    # If the snake is in the top section, generate sections downwards until the edge of the board.
    if pos_top:
        distance_to_wall = board_array[0] - generation_point[0] - 1
        for space in range(distance_to_wall):
            snake.append([generation_point[0] + space + 1, generation_point[1]])
            print("Added chunk...")
            print(snake)
            pieces_to_generate -= 1
            if pieces_to_generate == 0:
                break
    # Or if it's in the bottom section, generate upwards
    else:
        distance_to_wall = 0 + generation_point[0]
        for space in range(distance_to_wall):
            snake.append([generation_point[0] - space - 1, generation_point[1]])
            print("Added chunk...")
            print(snake)
            
            if pieces_to_generate == 0:
                break
            pieces_to_generate -= 1

    # Snake length is variable, so return if we're done
    if pieces_to_generate == 0:
                return snake

    # Otherwise, continue generation from the new tail
    generation_point = snake[-1]
    # If we're in the left section, generate right
    if pos_left:
        distance_to_wall = board_array[1] - generation_point[1] - 1
        for space in range(distance_to_wall):
            snake.append([generation_point[0], generation_point[1] + space + 1])
            print("Added chunk...")
            print(snake)
            pieces_to_generate -= 1
            if pieces_to_generate == 0:
                break
    # Or if we're in the right section, generate left
    else:
        distance_to_wall = 0 + generation_point[1]
        for space in range(distance_to_wall):
            snake.append([generation_point[0], generation_point[1] - space - 1])
            print("Added chunk...")
            print(snake)
            pieces_to_generate -= 1
            if pieces_to_generate == 0:
                break
    
    # Snake length is variable, so return if we're done
    if pieces_to_generate == 0:
                return snake
    # In most cases, going vertical and horizontal should be enough to generate the snake.
    # But for edge cases, we'll repeat the process, from here the snake would have generated into a corner.
    # This is a programming sin but it's a few hours in and I'm still generating the snake so below is the same code.
    generation_point = snake[-1]
    if generation_point[0] <= mid_y:
        pos_top = True
    else:
        pos_top = False
    
    if generation_point[1] <= mid_x:
        pos_left = True
    else:
        pos_left = False
    if pos_top:
        distance_to_wall = board_array[0] - generation_point[0] - 1
        for space in range(distance_to_wall):
            snake.append([generation_point[0] + space + 1, generation_point[1]])
            pieces_to_generate -= 1
            if pieces_to_generate == 0:
                break
    else:
        distance_to_wall = 0 + generation_point[0]
        for space in range(distance_to_wall):
            snake.append([generation_point[0] - space - 1, generation_point[1]])
            print("Added chunk...")
            print(snake)
            
            if pieces_to_generate == 0:
                break
            pieces_to_generate -= 1

    if pieces_to_generate == 0:
                return snake

    generation_point = snake[-1]
    if pos_left:
        distance_to_wall = board_array[1] - generation_point[1] - 1
        for space in range(distance_to_wall):
            snake.append([generation_point[0], generation_point[1] + space + 1])
            pieces_to_generate -= 1
            if pieces_to_generate == 0:
                break
    else:
        distance_to_wall = 0 + generation_point[1]
        for space in range(distance_to_wall):
            snake.append([generation_point[0], generation_point[1] - space - 1])
            pieces_to_generate -= 1
            if pieces_to_generate == 0:
                break
    

    print(snake)

    return snake

def maybe_plus_or_minus_one(value:int) -> int:
    option = random.randint(-1,1)
    value += option
    return value


    
generate_snake()

