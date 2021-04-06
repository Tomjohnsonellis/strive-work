"""
This is a dataset about the various attributes of breast cancer tumors.
This exercise will show some probabilities of tumors being either benign or malignant,
based on various attribute values.

For example:
What is the probability of this tumor being malignant if it has a large radius?
This tumor is very smooth, how likely is it to be benign?
This tumor has all the characteristics of being malignant, what are the chances that it isn't?

Bayes Theorem will be very helpful here.

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

# This dataset is in a format called "Bunch" which behaves a bit like a dictionary
columns = dataset.keys()
# for col in columns:
#     print(col)


# For this data analysis task, we are going to look at just the first 5 columns of data
# (Plus the diagnosis result)
chosen_columns = dataset.feature_names[0:6]
#print(chosen_columns)
first_five = []
for obs in dataset.data:
    first_five.append(obs[0:6])
# Construct a dataframe
df = pd.DataFrame(first_five, columns = chosen_columns)
#print(df)

# Malignant is 0, Benign is 1
#print(dataset.target_names)

# Add the diagnosis column to the data frame
df["diagnosis"] = dataset.target

# This data frame is 569 rows x 7 Columns
#print(f"Shape: {df.shape}")


# All Benign data
benigns = (df[df["diagnosis"] == 1])
# All Malignant data
malignants = (df[df["diagnosis"] == 0])

# print("Min Max Ben")
# print(min(benigns["mean radius"]))
# print(max(benigns["mean radius"]))
# print("Min Max Mal")
# print(min(malignants["mean radius"]))
# print(max(malignants["mean radius"]))
print("-----")
#print(benigns.describe())
#print(malignants.describe())
total_malignants = df["diagnosis"].value_counts()[0]
total_benigns = df["diagnosis"].value_counts()[1]
total_observations = total_benigns + total_malignants
# print(total_benigns)
# print(total_malignants)
# print(total_observations)

def nice_results(sample, total_data):
    return round(( (sample / total_data) * 100), 2)


# Overall probability of each category
print("For a tumor we know nothing about:")
print(f"Chance of benign: {nice_results(total_benigns, total_observations)}%")
print(f"Chance of malignant: {nice_results(total_malignants, total_observations)}%")

print("-----")
df.describe()
#print(benigns.describe())
#print(malignants.describe())


def greater_than_mean_plus_1sd(column_of_data):
    mean = column_of_data.mean()
    sd = column_of_data.std()
    # print(mean)
    # print(sd)
    #print(mean + sd)
    boolean_list = column_of_data.gt(mean + sd)
    #print(boolean_list)
    return boolean_list


def calculate_posterior(prob_hyp, prob_ev_giv_hyp, prob_ev_giv_not_hyp, prob_not_hyp):
    # Likelihood is: Prob(Evidence given Hypothesis)
    # Unconditional Probability is: ( Prob(Ev | Hyp) * Prob(Hyp) ) + ( Prob(Ev | ¬Hyp) * Prob(¬Hyp) )
    # P(H|E) = ( P(E|H) * P(H) )
    #           --------------
    #       ( P(E|H) * P(H) ) + (P(E|¬H) * P(¬H) )
    posterior = (prob_ev_giv_hyp * prob_hyp) / ((prob_ev_giv_hyp * prob_hyp) + (prob_ev_giv_not_hyp * prob_not_hyp))

    return posterior


# Let's start by asking some questions:
# "Given that this tumor is rather compact, how likely is it to be malignant?

def compact_malignant():
    # We'll use the bayesian method to answer this...

    # Hypothesis: This tumor is malignant
    PH = total_malignants / total_observations

    # We'll define "rather compact" as being at least one standard deviation above the average for tumors
    matching_data = greater_than_mean_plus_1sd(malignants["mean compactness"])
    # Evidence: mean compactness is > the mean + 1 standard deviation (0.157...)
    evidence = matching_data.value_counts()
    PE = evidence[1] / (evidence[0] + evidence[1])

    # P(Evidence | Hypothesis) = "The tumor is malignant, how likely is it to meet the criteria?"
    matching_data = greater_than_mean_plus_1sd(malignants["mean compactness"])
    EGH = matching_data.value_counts()
    PEGH = EGH[1] / (EGH[0] + EGH[1])

    # P(Evidence | ¬Hypothesis) = "The tumor is not malignant, how likely is it to meet the criteria?"
    matching_data = greater_than_mean_plus_1sd(benigns["mean compactness"])
    EGNH = matching_data.value_counts()
    PEGNH = EGNH[1] / (EGNH[0] + EGNH[1])
    # P(¬Hypothesis) = "This tumor is not malignant"
    # PNH = total_benigns / total_observations
    PNH = 1 - (total_malignants / total_observations)

    print("This tumor is rather compact, Is it malignant?")
    posterior = calculate_posterior(PH, PEGH, PEGNH, PNH)
    print(f"Probability of that being true: {round(posterior * 100, 2)}%")
    print(f"The evidence causes a {(posterior - PH) * 100}% change!")


compact_malignant()








