"""
A "pipeline" is a name for the processes you go through when solving problems, every type of work can have a pipeline,
not just ML. This file will have some notes on how to go about solving data science problems, as well as some useful
functions that I am likely to need regularly.

A typical workflow can go like this:

1. Business problem
    Your boss could say "Hey, we've got a question for you, how do we <keep our customers happy>?

2. Data Acquisition
    For whatever the problem is, think about what kind of data would be useful, what's critical and what's a bonus
    Where do you get it from? A database? Web scraping? Old paper files that need to be digitised? Whatever you can.

3. Data Preperation
    It's wishful thinking to presume that the data will be already nicely organised, consistent and complete.
    A lot of the grunt work as a data scientist is preparing/preprocessing the data. Get comfortable with it!
    This would involve doing stuff like one-hot encoding categories, judging if missing data is too problematic,
    imputing (filling missing values with appropriate dummies), amalgamating different data sources and making them
    all consistent. All the "admin" type work that when done well, makes the rest of the task go a lot more smoothly.

4. Data Analysis
    Now we get to be big-brain. Probably start with some exploratory data analysis (EDA) to play about with the data,
    make some trivial graphs, throw out a correlation matrix, just explore what you have, ponder the relationships,
    remember what it is you are actually trying to answer. This is more of a thinking procedure.
    It is important that you stay both logical and sensible: It's easy to just slap data together with no true insight.

5. Data modelling
    This is the fun part. This is where you already:
    1. Know what problem you're solving
    2. Have gathered data to help your solve it
    3. Have prepared the data so that you can use it effectively
    4. Have focused in on how you will use that data
    Now you build models, use math, use algorithms, make graphs, produce insights, and answer the problem.

6. Visualisation and Communication
    Now that you have used your skills as a data scientist to deeply understand and solve the problem you had,
    it's time to share that knowledge with the world. Make things pretty, clear, and explain what you have found.
    You are likely to be presenting these findings to non-technical people, those who need your skills and know-how.
    There's no point in solving problems and then having the solution be incomprehensible, you should be able to
    explain it to anyone who is interested, no matter their understanding of your work.

7. Deployment and maintenance
    After a write-up or presentation, it will be time to wrap up the project and go on to the next one.
    But it is important to leave your findings in excellent condition for the next people who may be interested.
    Your solution may prove useful in later projects and you will thank yourself for leaving it in such a good state.
    A streamlit website, github readme, notes and documentation, interactive graphs, whatever you want really!

And that's a basic outline of how a project could go - it's good to break down projects into steps, and sub-steps.
----------
Now for a machine learning pipeline, a helpful function that will allow you to work more effectively.
"""
# Typical imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# What a pipeline function should do is: take some data and produce some results from it
# It's just for convenience so that we don't have to redo basic things every time we want to do some work.


# Use good ol' iris data for an example
some_dataset = datasets.load_iris()

print(some_dataset.keys())

# Whatever we're doing, as long as we're trying to predict something we will need...
# X: Data to use for predictions
# y: What we want to predict
X = some_dataset.data
y = some_dataset.target


def split_data(X,y,train_size=0.8, random_state=0):
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
    print("Data has been split in the standard way, created global variables: X_train, X_test, y_train, y_test")
    return

def scale_data(X_train, X_test, normalise=False):
    if normalise:
        X_train = preprocessing.normalize(X_train)
        X_test = preprocessing.normalize(X_test)
        print("Data has been normalised")
    else:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform((X_test))
        print("Data has been scaled with StandardScaler")


def train_model(X_train, y_train):
    global clf

    models = {
        1: "Logistic Regression",
        2: "K-Means",
        3: "Random Forest"
    }
    print("Choose a model!")
    for key in models:
        print(f"{key} : {models[key]}")
    choice = int(input(">>"))

    if choice == 1:
        # Logistic regression
        clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    elif choice == 2:
        clf = KMeans().fit(X_train, y_train)
    elif choice == 3:
        clf = RandomForestClassifier().fit(X_train, y_train)


    return clf


def test_model(clf, X_test, y_test):
    print(f"Model: {clf}")
    print(f"Score: {clf.score(X_test, y_test) *100}%")
    return


def show_cv_results(clf, X, y, graphs=False):
    print(f"Cross validating using: {clf}...")
    cv = cross_validate(clf, X, y)
    for test, score in enumerate(cv["test_score"]):
        print(f"Test {test} - {round(score*100,2) }%" )
    print(f"Average accuracy: {round(cv['test_score'].mean() *100, 2)}%")
    if graphs:
        plt.bar([1,2,3,4,5], cv["test_score"]*100)
        plt.plot([1,5],[cv['test_score'].mean()*100,cv['test_score'].mean()*100], color="red",linewidth=3, linestyle=":", label=f"Average: {round(cv['test_score'].mean() *100, 2)}%")
        plt.xlabel("Test Number")
        plt.ylabel("Accuracy")
        plt.title("Cross Validation Results")
        plt.legend()
        plt.show()

def basic_pipeline(X,y, normalise=False):
    split_data(X, y, train_size=0.5)
    scale_data(X_train, X_test, normalise=normalise)
    train_model(X_train, y_train)
    return clf


if __name__ == '__main__':
    split_data(X,y, train_size=0.5)
    scale_data(X_train, X_test, normalise=True)
    train_model(X_train, y_train)
    test_model(clf, X_test, y_test)
    show_cv_results(clf, X, y)
