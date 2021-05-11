import optuna
import sklearn.metrics
from optuna.samplers import TPESampler
from sklearn import pipeline, svm, compose
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

data = load_diabetes()
print(data.keys())
columns = data.feature_names
X = pd.DataFrame(data = data.data, columns=columns)
y = data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# All the data is numerical
num_vars = columns

num_preprocessing = pipeline.Pipeline(steps=[
    ('scaler', preprocessing.StandardScaler())
])

# cat_preporcessing = pipeline.Pipeline(steps=[
#     ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore'))
# ])

prepro = compose.ColumnTransformer(transformers=[
    ('num', num_preprocessing, num_vars),
    #('cat', cat_preporcessing, cat_vars),
]) # Drop other vars not specified in num_vars or cat_vars
#
# X_train = prepro.fit_transform(X_train)
# X_test = prepro.transform(X_test)

# print(y_train.mean())
# print(y_test.mean())
def objective(trial):

    model_name = trial.suggest_categorical("Model Name: ", ["Lassoo", "Bayesian", "RandomForest"])

    if model_name == "Lassoo":
        alpha_tests = trial.suggest_uniform(name="alpha", low=0.0, high=3.0)
        fit_tests = trial.suggest_categorical("fit_intercept", [False, True])
        model = linear_model.Lasso(alpha=alpha_tests, fit_intercept=fit_tests)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error = sklearn.metrics.mean_squared_error(y_test, y_pred)
        return error

    if model_name == "Bayesian":
        alpha_1_tests = trial.suggest_uniform(name="alpha_1", low=0.001, high=0.1)
        alpha_2_tests = trial.suggest_uniform(name="alpha_2", low=0.001, high=0.1)
        lambda_1_tests = trial.suggest_uniform(name="lambda_1", low=0.0001, high=0.01)
        lambda_2_tests = trial.suggest_uniform(name="lambda_2", low=0.0001, high=0.01)
        model = linear_model.BayesianRidge(
            alpha_1=alpha_1_tests,
            alpha_2=alpha_2_tests,
            lambda_1=lambda_1_tests,
            lambda_2=lambda_2_tests
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error = sklearn.metrics.mean_squared_error(y_test, y_pred)
        return error

    if model_name == "RandomForest":
        n_est_tests = int(trial.suggest_uniform(name="n=", low=1, high=20))
        depth_tests = int(trial.suggest_uniform(name="Max Depth: ", low=2, high=5))
        feat_tests = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", None])
        model = RandomForestRegressor(
            n_estimators=n_est_tests,
            max_depth=depth_tests,
            max_features=feat_tests
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error = sklearn.metrics.mean_squared_error(y_test, y_pred)
        return error

    if model_name == "Linear_reg":
        model = linear_model.LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error = sklearn.metrics.mean_squared_error(y_test, y_pred)
        return error


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)









# def objective(trial):
#     x = trial.suggest_uniform("x", -3, 3)
#     return (x + 1) ** 2
#
# # study = optuna.create_study()
# # study.optimize(objective, n_trials=100)
# #
# # study.best_params
#
#
# # TPEsampler
#
# study = optuna.create_study(direction="minimize", sampler=TPESampler())
# study.optimize(objective, n_trials=10)
#
# def objective(trial):
#     #fit_intercept = trial.suggest_categorial("fit_intercept", [False, True])
#     c = trial.suggest_loguniform("svc_c", 1e-10, 1e10)
#     kernel = trial.suggest_categorial("kernel", ["linear", "poly", ""])
#     kernel = trial.suggest_linear
#
#     model = svm.SVR(C=c, kernel=kernel, gamma="auto")




#numerical_prepro = pipeline.Pipeline()
