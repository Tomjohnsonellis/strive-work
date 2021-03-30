import numpy as np
import pandas as pd

# A linear combination is all the possible vectors that can be reached by scaling a combination of vectors
# If you have 2 vectors, as long as they aren't the same line, you can reach any point in 2d space
# If they are on the same line, all you can really do is move along that line.
# Similar for 3 vectors, if they are all at least slightly different, you can access any point in 3d space.
# If the 3rd dimension is the same across all vectors, you would make a plane in 3d space

# A redundant vector in this case would be Linearly dependant, it can be expressed as a linear combination of another
# Vectors that do add another dimension Linearly independant


# I'm not sure I understand the question, I've gone with:
# "Which of the given vectors are linear combinations of another?"

def plane(a ,b ,c):
    # returns false if the vectors don't define a plane and returns which are those vectors if true

    # This was challenging more for finding the syntax than solving the problem, I went with "if a vector mod vector results in all 0s, it's a factor"

    if np.all(np.mod(a ,b) == 0) or (np.all(np.mod(b ,a) == 0)):
        print(f"{a} and {b} are linear combinations")
        return[a ,b]
    if (np.all(np.mod(a ,c) == 0)) or (np.all(np.mod(c ,a) == 0)):
        print(f"{a} and {c} are linear combinations")
        return[a ,c]
    if (np.all(np.mod(b ,c) == 0) ) or (np.all(np.mod(c ,b) == 0)):
        print(f"{b} and {c} are linear combinations")
        return[b ,c]

    return False


v1 = [1,2,3]
v2 = [2,2,2]
v3 = [8,8,8]

plane(v1,v2,v3)
