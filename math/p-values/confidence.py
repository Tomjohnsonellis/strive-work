import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

"""
I had some difficulty grasping confidence intervals, so I'm going to make a simulation of them

Roll 100 dice, plot the results
Calculate the 95% confidence interval for those results
Roll 100 rigged dice, add the mean of that to the plot
See how well the confidence interval holds up

Note to peer reviewers: This was just me getting an understanding of how a confidence interval works,
It is not the exercises for the day
"""


def roll_dice(amount):
    results = []
    while amount > 0:
        dice1 = random.randint(1, 6)
        dice2 = random.randint(1, 6)
        total = dice1 + dice2
        results.append(total)
        amount -= 1
    return results


def roll_magic_dice(amount):
    results = []
    while amount > 0:
        dice1 = random.randint(2, 6)
        dice2 = random.randint(2, 6)
        total = dice1 + dice2
        results.append(total)
        amount -= 1
    return results


def repeated_test():
    test_data = roll_dice(dice_to_roll)
    test_data = pd.DataFrame([test_data]).transpose()
    test_results = test_data.value_counts().sort_index()
    plt.scatter(dice_range, test_results)
    return test_results


def initial_graph():
    initial_data = roll_dice(dice_to_roll)
    initial_data = pd.DataFrame([initial_data]).transpose()
    sorted_initial = initial_data.value_counts().sort_index()
    print(type(sorted_initial))
    print(sorted_initial)
    plt.bar([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], sorted_initial.values, alpha=0.75, color="green")
    # Stats
    initial_stats = {
        "mean": np.mean(initial_data[0]),
        "sum": np.sum(initial_data[0]),
        "n": dice_to_roll,
        "std": np.std(initial_data[0])
    }
    return initial_stats


def magic_graph():
    magic_data = roll_magic_dice(dice_to_roll)
    magic_data = pd.DataFrame([magic_data]).transpose()
    sorted_magic = magic_data.value_counts().sort_index()
    print(sorted_magic)
    plt.bar([4, 5, 6, 7, 8, 9, 10, 11, 12], sorted_magic.values, alpha=0.5, color="red")
    # Magic Stats
    magic_stats = {
        "mean": np.mean(magic_data[0]),
        "sum": np.sum(magic_data[0]),
        "n": dice_to_roll,
        "std": np.std(magic_data[0])
    }
    return magic_stats


def report(initial, magic):
    # For 95% confidence
    z = 1.96

    # Standard Errors
    initial["std_error"] = initial["std"] / np.sqrt(initial["n"])
    magic["std_error"] = magic["std"] / np.sqrt(magic["n"])

    initial["upper_conf"] = initial["mean"] + (z * initial["std_error"])
    initial["lower_conf"] = initial["mean"] - (z * initial["std_error"])

    # magic["upper_conf"] = magic["mean"] + (z * initiial["std_error"])
    # magic["lower_conf"] = magic["mean"] - (z * initiial["std_error"])

    biggest_diff = initial["upper_conf"] - initial["lower_conf"]
    difference_in_mean = abs(initial["mean"] - magic["mean"])
    # print(difference_in_mean)
    print("-------------------------")
    print("Are those dice rigged?")
    print("-------------------------")
    print("For a nice fair dice, we would expect that:")
    print("The mean is about: {}".format(initial["mean"]))
    print("At most it could be: {}".format(initial["upper_conf"]))
    print("At least it could be: {}".format(initial["lower_conf"]))
    print("It can vary by up to this much: {}".format(biggest_diff))
    print("-------------------------")
    print("Let's compare to the other dice")
    print("-------------------------")
    print("Magic dice mean: {}".format(magic["mean"]))
    print("Are we confident this is normal?")
    print("The biggest difference in the mean we will accept is: {}".format(biggest_diff))
    print("This difference is: {}".format(difference_in_mean))
    above = (magic["mean"] > initial["upper_conf"])
    below = (magic["mean"] < initial["lower_conf"])
    print("Above our highest value?: {}".format(above))
    print("Below our lowest value?: {}".format(below))
    normal = not bool(above + below)
    if not normal:
        print("We can safely say, with 95% confidence, that these dice are not fair")
        print("We will REJECT the null hypothesis (The dice are normal")
    else:
        print("We can safely say we're just a sore loser.")

    plt.scatter(initial["mean"], 175, color="green", label="Fair Mean")
    plt.scatter(magic["mean"], 175, color="red", label="Other mean")
    plt.plot([initial["mean"], magic["mean"]], [175,175],color="black",linewidth=2, label="Difference in means")
    plt.xlabel("Results of throwing 2 Dice")
    plt.ylabel("Quantity of result")
    plt.title("Are those dice rigged?")



retries = 10
dice_to_roll = 1000

dice_range = []
for integer in (range(2, 13, 1)):
    dice_range.append(integer)

fair = initial_graph()
rigged = magic_graph()
report(fair, rigged)
plt.legend()
plt.show()
