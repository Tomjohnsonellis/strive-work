"""
This is my attempt at the kaggle titanic dataset, the goal is to predict if people will survive the titanic!
"Women and children first!"
https://www.kaggle.com/c/titanic/overview

This was done in a team of 3 with: Fabio Fistarol and Hedaya Ali
We used some feature engineering strategies to get more use out of the data,
and tried various machine learning models to see what worked best.
"""

# The usual imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Model stuff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score


# Feature engineering

#years = X_train.Age.astype(int).value_counts().sort_index()
#plt.scatter(years.index, years.values)
#years
#years = years.transpose()
# baby = 0-4
# child = 5-9
# teen = 10-14
# young_adult = 15-19
# adult = 20-40
# senior = 41+

def bin_ages(X_train):
    X_train["Age"] = X_train["Age"].astype(int)
    X_train["baby"] = X_train["Age"].le(4).astype(int)
    X_train["child"] = X_train["Age"].between(5,9).astype(int)
    X_train["teen"] = X_train["Age"].between(10,14).astype(int)
    X_train["young_adult"] = X_train["Age"].between(15,19).astype(int)
    X_train["adult"] = X_train["Age"].between(20,40).astype(int)
    X_train["senior"] = X_train["Age"].ge(41).astype(int)
    return X_train


def one_hot_embark(X):
    one_hots = pd.get_dummies(X["Embarked"], prefix='Embark')
    X["Embark_C"] = one_hots["Embark_C"]
    X["Embark_Q"] = one_hots["Embark_Q"]
    X["Embark_S"] = one_hots["Embark_S"]
    return X


def one_hot_sex(X):
    dummies = pd.get_dummies(X.Sex)
    X["female"] = dummies["female"]
    X["male"] = dummies["male"]
    return X


def one_hot_class(X):
    class_dummies = pd.get_dummies(X.Pclass)
    X["Upper_class"] = class_dummies[1]
    X["Middle_class"] = class_dummies[2]
    X["Lower_class"] = class_dummies[3]
    return X




# Load the data
train = pd.read_csv("train.csv")
# Turn Male/Female into 1/0
#train.Sex = train.Sex.eq("male").astype(int)
# one-hot encode them



# Clean up some incomplete rows
train = train.dropna(subset=["Age"])
train = train.dropna(subset=["Embarked"])

y = train.Survived
X = train.drop(["Survived"], axis=1)
X.index = X.PassengerId


X = one_hot_embark(X)
X = one_hot_sex(X)
X = one_hot_class(X)
X = bin_ages(X)

X = X.drop(["PassengerId", "Name","Age", "Sex","Cabin", "Ticket", "Embarked"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

clf = RandomForestClassifier(criterion="entropy", random_state=0, n_estimators=100)
clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_test, y_test)
print(f"Random Forest scored: {scores.mean()}")
