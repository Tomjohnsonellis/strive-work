"""
Snake pathfinder task

Given:
A board - b,
A snake - s,
A depth - d,
How many valid paths of length d are available?

Constraints:
1 <= depth <= 20

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

# board = [4,5]
# snake_length = 5
# print(f"Board valid?: {validate_board(board)}")

def generate_snake(snake_length:int=5, board_array:list[int,int]=[4,5]) -> list[list[int,int]]:
    y = random.randint(0, board_array[0])
    x = random.randint(0, board_array[1])
    snake_head = [y,x]


    snake = [snake_head]
    snake = create_snake_body(snake, snake_length, board_array)



    return snake

def create_snake_body(snake:list[list[int,int]], snake_length:int, board_array:list[int,int]=[4,5]) -> list[list[int,int]]:
    board = board_array
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

    # # TESTING
    # print(f"MID POINT: {mid_y, mid_x}")
    # print(f"SNAKE HEAD: {snake[0]}")
    # print(f"TOP: {pos_top}")
    # print(f"LEFT: {pos_left}")


    # If we're in the left section, generate right
    if pos_left:
        generate_right(snake, board)
    # Or if we're in the right section, generate left
    else:
        generate_left(snake, board)
    
    # If the snake is in the top section, generate sections downwards until the edge of the board.
    if pos_top:
        generate_down(snake, board)
    # Or if it's in the bottom section, generate upwards
    else:
        generate_up(snake, board)

    # Snake length is variable, originally I was constantly checking to see if we had generated enough pieces
    # But later decided that it's far simpler to just generate more then cut the snake down to size.
    # In most cases, going vertical and horizontal should be enough to generate the snake.
    # But for edge cases, we'll repeat the process, from here the snake would have generated into a corner.
    if snake[-1][0] <= mid_y:
        pos_top = True
    else:
        pos_top = False
    
    if snake[-1][1] <= mid_x:
        pos_left = True
    else:
        pos_left = False
    if pos_left:
        generate_right(snake, board)
    else:
        generate_left(snake, board)
    if pos_top:
        generate_down(snake, board)
    else:
        generate_up(snake, board)

    # Snip the snake
    snake = snake[0:snake_length]

    return snake


def generate_down(snake, board):
    tail = snake[-1]
    distance_to_wall = board[0] - tail[0] - 1
    
    if distance_to_wall <= 0:
        print("Nowhere to go!")
        return snake
    for space in range(distance_to_wall):
        snake.append([ tail[0] + space + 1, tail[1] ])
    return snake


def generate_up(snake, board):
    tail = snake[-1]
    distance_to_wall = 0 + tail[0]
    
    if distance_to_wall <= 0:
        print("Nowhere to go!")
        return snake
    for space in range(distance_to_wall):
        snake.append([ tail[0] - (space + 1), tail[1] ])
    return snake


def generate_left(snake, board):
    tail = snake[-1]
    distance_to_wall = 0 + tail[1]
    if distance_to_wall <= 0:
        print("Nowhere to go!")
        return snake
    
    for space in range(distance_to_wall):
        snake.append([ tail[0], tail[1] - (space + 1)])
    return snake

def generate_right(snake, board):
    tail = snake[-1]
    distance_to_wall = board[1] - tail[1] - 1
    if distance_to_wall <= 0:
        print("Nowhere to go!")
        return snake

    for space in range(distance_to_wall):
        snake.append([ tail[0], tail[1] + (space + 1)])
    return snake

if __name__ == "__main__":
    print("Use generate_snake(length, board) to creat a snake.")

