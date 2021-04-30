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
def feature_engineering(X):
    X = one_hot_embark(X)
    X = one_hot_sex(X)
    X = one_hot_class(X)
    X = bin_ages(X)
    return X


def bin_ages(X_train):
    # Change the ages data from an age to age range
    # baby = 0-4
    # child = 5-9
    # teen = 10-14
    # young_adult = 15-19
    # adult = 20-40
    # senior = 41+
    X_train["Age"] = X_train["Age"].astype(int)
    X_train["baby"] = X_train["Age"].le(4).astype(int)
    X_train["child"] = X_train["Age"].between(5, 9).astype(int)
    X_train["teen"] = X_train["Age"].between(10, 14).astype(int)
    X_train["young_adult"] = X_train["Age"].between(15, 19).astype(int)
    X_train["adult"] = X_train["Age"].between(20, 40).astype(int)
    X_train["senior"] = X_train["Age"].ge(41).astype(int)
    return X_train


def one_hot_embark(X):
    # One-hot encode the place people embarked from
    one_hots = pd.get_dummies(X["Embarked"], prefix='Embark')
    X["Embark_C"] = one_hots["Embark_C"]
    X["Embark_Q"] = one_hots["Embark_Q"]
    X["Embark_S"] = one_hots["Embark_S"]
    return X


def one_hot_sex(X):
    # One-hot encode the sex of passengers
    one_hots = pd.get_dummies(X.Sex)
    X["female"] = one_hots["female"]
    X["male"] = one_hots["male"]
    return X


def one_hot_class(X):
    # One-hot encode the socio-economic class of each passenger
    one_hots = pd.get_dummies(X.Pclass)
    X["Upper_class"] = one_hots[1]
    X["Middle_class"] = one_hots[2]
    X["Lower_class"] = one_hots[3]
    return X


def cleanup(raw_data):
    # Clean up some incomplete rows
    raw_data = raw_data.dropna(subset=["Age"])
    raw_data = raw_data.dropna(subset=["Embarked"])
    raw_data = raw_data.dropna(subset=["Fare"])
    return raw_data


def remove_excess(X):
    X = X.drop(["PassengerId", "Name", "Age", "Sex", "Cabin", "Ticket", "Embarked"], axis=1)
    return X


def build_model():
    # Load the data
    train = pd.read_csv("train.csv")
    # Clean up some incomplete rows
    train = cleanup(train)

    # Determine the target(y) and data(X) to use
    y = train.Survived
    X = train.drop(["Survived"], axis=1)
    # Make the index passengerId so we can keep it but not have it influence the model
    X.index = X.PassengerId

    # Do some data manipulation
    X = feature_engineering(X)
    # Drop anything we aren't using or don't think is useful
    X = remove_excess(X)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
    # Make and fit a model
    clf = RandomForestClassifier(criterion="entropy", random_state=0, n_estimators=1000)
    clf.fit(X_train, y_train)
    return clf


def prep_test():
    test_data = pd.read_csv("test.csv")
    test_data["Age"] = test_data["Age"].fillna(test_data["Age"].mean())
    test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].mean())
    test_data = feature_engineering(test_data)
    test_data.index = test_data.PassengerId
    test_data = remove_excess(test_data)
    return test_data


def create_submission(test_data, clf):
    kaggle_predictions = clf.predict(test_data)
    pid = test_data.index
    columns = ["PassengerId", "Survived"]
    df = pd.DataFrame(kaggle_predictions)
    df.index = pid
    df.columns = ["Survived"]
    df.to_csv("kaggle-predictions.csv")
    print("Done.")
    return


if __name__ == '__main__':
    clf = build_model()
    test_data = prep_test()
    create_submission(test_data, clf)
