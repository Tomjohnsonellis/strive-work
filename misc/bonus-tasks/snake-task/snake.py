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
board = [3,5]
print(f"Board valid?: {validate_board(board)}")

def generate_snake(snake_length:int=4, board_array:list[int,int]=board) -> list[list[int,int]]:
    y = random.randint(0, board_array[0])
    x = random.randint(0, board_array[1])
    snake_head = [y,x]
    snake = [snake_head]

    # Create X blocks
    # Use the length and just create all the blocks at once



    return snake

def create_snake_body(snake:list[list[int,int]], board_array:list[int,int]=board) -> list[list[int,int]]:
    # The idea here is to build the snake in a way that avoids the snake extending itself into a corner
    mid_y = board_array[0]/2
    mid_x = board_array[1]/2
    # We'll split the board into 4 sections, and the section that the snake head is in determines how it will be extended
    head_location = ""
    if snake[0][0] < 




    return snake

def maybe_plus_or_minus_one(value:int) -> int:
    option = random.randint(-1,1)
    value += option
    return value


    
generate_snake()

