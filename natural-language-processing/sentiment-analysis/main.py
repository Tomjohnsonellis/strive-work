import pandas as pd
import numpy as np
import torchtext

amazon = torchtext.datasets.AmazonReviewFull
training_iterator = amazon(split="test") # The training dataset is massive, the test will be enough for us

def tokenize(label, line):
    return line.split()


count = 0
for label, line in training_iterator:
    print(label, " )-( ", line)
    count += 1
    if count >= 10:
        break


dataset = []
for label, line in training_iterator:
    tokens = []
    tokens += tokenize(label, line)
    

print("-"*50)
print(len(tokens))
print(tokens[0])