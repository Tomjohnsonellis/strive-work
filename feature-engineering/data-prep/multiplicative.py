import numpy    as np
import pandas   as pd
import seaborn  as sb
import matplotlib.pyplot as plt
import sklearn  as skl

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose, tree, svm, ensemble, gaussian_process, neighbors
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import linear_model    # LogisticRegression
from sklearn import set_config

"""
Task today is to build a robust pipeline similar to yesterday,
BUT for multiplicative classifiers!


"""
# When working with any kind of data set, you will need to understand your data VERY well.
# At the very least we need to know if data is numerical or categorical
# For the sake of pipeline usage, you can keep lists of each type (These would be the column names)
categorical_values = ["Gender", "Member of a club", "Dominant Hand", "Cars Owned"]
numerical_values = ["Age", "Income", "No. of burgers eaten annually"]

# You can then pass this info to a pipeline, which typically will do different things to different types of data
numerical_preprocessing = pipeline.Pipeline(steps=[
    # Imputing does require some sense, you might want to construct your own imputer function!
    # For example you may wish to use the average stat of only data that matches several criteria
    ("imputer", impute.SimpleImputer(strategy="constant", fill_value="-1")),
    # You may want to normalise or scale, depending on what's best for the data!
    ("normalizer", preprocessing.Normalizer())
])

categorical_preprocessing = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='most frequent', add_indicator=True)),
    # One Hot encoding is the goto for most numerical categories
    ("onehot", preprocessing.OneHotEncoder(handle_unknown="ignore"))
])

multiplicative_model_preprocessing = compose.ColumnTransformer(transformers=[
    ("numbers", numerical_preprocessing, numerical_values), # Do the numerical preprocessing on the numerical data
    ("categories", categorical_preprocessing, categorical_values)
], remainder="drop") # Drop anything that we've decided not to use

# See what it looks like in a notebook
multiplicative_model_preprocessing


multiplicative_classifiers ={
    #Dictionary of classifiers
    "SVM": svm.LinearSVR(),
    "Regression Tree": tree.DecisionTreeRegressor(),
    "GB Regressor": ensemble.GradientBoostingRegressor(),
    "Gaussian Regressor": gaussian_process.GaussianProcessRegressor(),
    "KNN": neighbors.KNeighborsRegressor()
}

multiplicative_pipelines = {name: pipeline.make_pipeline(multiplicative_model_preprocessing, model) for name, model in
                            multiplicative_classifiers.items()}


"""
Could try to use this on some data!
Do need to understand what everything does though.
"""




"""
Example custom imputer, groups things and fills blanks with the average of the groups
==========================

# We choose some groups that we think matter in our particular dataset
group_columns = ['Gender','Member of a club', "Cars Owned"]

# This finds the average age of people who are in those specific groups
imputation_map = df.groupby(group_columns)["Cars Owned"].mean().reset_index(drop=False)

for index, row in imputation_map.iterrows(): # Iterate through all possible group combinations 

    # A Boolean column, true when the sample meets our criteria (In these specific groups)
    indexes_of_matches = (df[group_cols] == row[group_cols]).all(axis=1) # Returns Boolean column with the length of dataframe  
          
    # Replace any missing values with the average age of matching data
    df[indexes_of_matches] = df[indexes_of_matches].fillna(row["Cars Owned"])
    

"""

#
# numerical_prep = pipeline.Pipeline(steps=[
#     ("imputer", impute.SimpleImputer(strategy="constant", fill_value=42069)),
#     ("normalizer", preprocessing.Normalize() )
# ])
#
# categorical_prep = pipeline.Pipeline(steps=[
#     imputer
#     ("onehot encoder", preprocessing.OneHotEncoder(handle_unknown="error"))
# ])
#

#
