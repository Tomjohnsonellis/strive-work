import numpy as np

# Harder
# Can you build a matrix solver that can do output the result given n matrices and a sequence of operations such as [add subtract multiply transpose?]
#Some ideas on valid operations: Multiplication Addition Subtraction Inverstion Transposed.

"""
# No I cannot, luckily some brain-boxes can!
# Explanation https://stackabuse.com/solving-systems-of-linear-equations-with-pythons-numpy/
# Basically, if you have some equations like:
# 4x + 2y = 7
# 33x + 88y = 666
# You can pass the coeffecients and results into .solve as lists and it will work it out.
"""
a = [ [4,2], [33,88] ]
b = [7, 666]
print( np.linalg.solve(a,b) )
