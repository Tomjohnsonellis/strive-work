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
chosen_columns = dataset.feature_names[0:5]
#print(chosen_columns)
first_five = []
for obs in dataset.data:
    first_five.append(obs[0:5])
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
print("##### ALL DATA #####")
print(np.mean(df[:]))
print("===== BENIGNS =====")
print(np.mean(benigns[:]))
print("!!!!! MALIGNANTS !!!!!")
print(np.mean(malignants))
print("-----")
print((np.mean(malignants[:]) - np.mean(benigns[:])))
print(0.5 * (np.mean(malignants[:]) - np.mean(benigns[:])))

# From this, we can see that malignants have a higher value (on average) in all categories
# So to reduce each data point down to a boolean, we will see which average the data is closer to
# False (0) for malignant, True (1) for benign

# To calculate which average a value is closer to...
def closer_to(dataframe, benign_mean, malignant_mean):
    closer_df = dataframe
    midpoints = (0.5 * (malignant_mean - benign_mean))
    #print(midpoints)
    threshold = (benign_mean + midpoints)
    # If it's above the midpoint, True
    truth_table = (np.greater(closer_df, threshold))
    # The "average" diagnosis is awkward here, I'm just going to drop it and re-add the correct one
    truth_table = truth_table.drop(["diagnosis"], axis=1)
    #truth_table["diagnosis"] = dataframe["diagnosis"]
    #print(truth_table)

    return truth_table


boolean_table = closer_to(df, np.mean(benigns), np.mean(malignants))
boolean_table
boolean_table_score = np.sum(boolean_table, axis=1)
score_probs = pd.DataFrame([df.diagnosis, boolean_table_score], index=["diagnosis","score"]).transpose()
#print(boolean_table.columns)
#print(score_probs)

malignant_scores = (score_probs[score_probs["diagnosis"] == 0]).drop(["diagnosis"], axis=1)
benign_scores = (score_probs[score_probs["diagnosis"] == 1]).drop(["diagnosis"], axis=1)

#print(malignant_scores)
#print(malignant_scores.value_counts())

# Here's a graph showing that malignant values are more likely to occur
# if you have more values closer to the average for malignants (Groundbreaking I know)
plt.hist(benign_scores, 4, alpha = 0.5, color="green")
plt.hist(malignant_scores, 4,alpha = 0.5, color="red")
plt.show()

