<h1 align="center">Day 2: Data Cleaning (Missings and Outliers)</h1>

## Exercises

### ❓ Missing values

1. What is the missing datatype used in pandas?
   <br>> np.nan
2. How to replace all occurences of the value 9999 to missing in pandas?
   <br>> pd.DataFrame.replace(9999, np.nan)
   <br>> or: df.replace(9999, np.nan)
3. How to get the absolute number of missings for each variable in pandas?
   <br>> df.isnull.sum()
4. How to get the percentage of missings for each variable in pandas?
   <br>> df.isnull.sum()/len(df)
5. How to drop rows with missing values?
   <br>> df.dropna()
6. How to drop variables with missing values?
   <br>> df.dropna(axis="columns")
7. What is the univariate imputation method in sklearn?
   <br>> impute.SimpleImputer()
8. What is the multivariate imputation method in sklearn?
   <br>> impute.IterativeImputer
9. What is the best univariate imputation method to categorical variables? (Explain why)
   <br>> "Best" depends on what you are aiming for, the methods range from doing nothing all the way to
   <br>> training deep learning models to find the best fit.
   <br>> A simple method is filling with the most common value, but this can introduce bias 
   <br>> as it makes it even more common.
10. What is the best univariate imputation method to numerical variables? (Explain why)
   <br>> Again, many methods exist and the "best" one will depend on your dataset and what you are doing.
    <br>> Filling with the mean is a popular simple method, as well as filling with some constant value like 0 or 9999
    <br>> Other methods such as filling with a random sample from the dataset, or the fancier
    <br>> random sample from a normalised distribution of the data constructed with the mean and variance of the dataset

### 🔎 Outliers

1. What is an outlier?
   <br>> A data point that SIGNIFICANTLY differs from the other points in the sample, e.g. it's 50x bigger
2. What is a simple method to detect and deal with outliers of a numerical variable?
   <br>> Luckily we have a mathematical definition of both minor and major outliers thanks to statistics.
   <br>> Typically a minor outlier is more than 1.5x the InterQuartileRange away from the mean
   <br>> And a major outlier is over 3x the IQR away
3. What is novelty detection?
   <br>> The process of detecting "novelties", which means outliers, anomalies, any kind of unusual data
   <br>> ML algorithms tend to be used for this and they can help improve the data for other models! [Link](https://deepai.org/machine-learning-glossary-and-terms/novelty-detection)
4. Name 4 advanced methods of outlier detection in sklearn.
   <br>> [Docs](https://scikit-learn.org/stable/modules/outlier_detection.html#overview-of-outlier-detection-methods)
   <br>> Robust Covariance, One-Class SupportVectorMachine, Isolation Forest, Local Outlier Factor

### 🖋 Typos

1. What is a typo?
   <br>> Definition: A typographical error
   <br>> wehn you misstyp somthing!
2. What is a good method of automatically detect typos?
<br>> Use a [list](https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines) of common misspellings and automatically correct them
<br>> [Bonus](https://stackoverflow.com/questions/487003/how-to-detect-a-typo-in-a-product-search-and-suggest-possible-corrections): Use one of many algorithms available for the task

### Practical case

Consider the following dataset: [San Francisco Building Permits](https://www.kaggle.com/aparnashastry/building-permit-applications-data). Look at the columns "Street Number Suffix" and "Zipcode". Both of these contain missing values.

- Which, if either, are missing because they don't exist?
  <br>> It is likely that "Street Number Suffix" would be missing due to not existing, not every building is apartments.
- Which, if either, are missing because they weren't recorded?
<br>> "Zipcode" is probably just not recorded as it's the American equivalent of a post code, everywhere has one for mail purposes.
  
Hint: Do all addresses generally have a street number suffix? Do all addresses generally have a zipcode?



| Var # |  NaN % | Var name                               | Var Description                                    |
|------:|-------:|:---------------------------------------|:---------------------------------------------------|
|     1 |      0 | Permit Number                          | Number assigned while filing                       |
|     2 |      0 | Permit Type                            | Type of the permit represented numerically.        |
|     3 |      0 | Permit Type Definition    | Description of the Permit type, for example new construction, alterations |
|     4 |      0 | Permit Creation Date      | Date on which permit created, later than or same as filing date           |
|     5 |      0 | Block                                  | Related to address                                 |
|     6 |      0 | Lot                                    | Related to address                                 |
|     7 |      0 | Street Number                          | Related to address                                 |
|     8 | 98.885 | **Street Number Suffix**               | Related to address                                 |
|     9 |      0 | Street Name                            | Related to address                                 |
|    10 |  1.391 | Street Name Suffix                     | Related to address                                 |
|    11 | 85.178 | Unit                                   | Unit of a building                                 |
|    12 | 99.014 | Unit suffix                            | Suffix if any, for the unit                        |
|    13 |  0.145 | Description         | Details about purpose of the permit. Example: reroofing, bathroom renovation     |
|    14 |      0 | Current Status                         | Current status of the permit application.          |
|    15 |      0 | Current Status Date                    | Date at which current status was entered           |
|    16 |      0 | Filed Date                             | Filed date for the permit                          |
|    17 |  7.511 | Issued Date                            | Issued date for the permit                         |
|    18 | 51.135 | Completed Date  | The date on which project was completed, applicable if Current Status = “completed”   |
|    19 |  7.514 | First Construction Document Date       | Date on which construction was documented          |
|    20 | 96.519 | Structural Notification                | Notification to meet some legal need, given or not |
|    21 | 21.510 | Number of Existing Stories | Num of existing stories in the building. Not applicable for certain permit types|
|    22 | 21.552 | Number of Proposed Stories             | Number of proposed stories for the construction/alteration    |
|    23 | 99.982 | Voluntary Soft-Story Retrofit          | Soft story to meet earth quake regulations      |
|    24 | 90.534 | Fire Only Permit                       | Fire hazard prevention related permit           |
|    25 | 26.083 | Permit Expiration Date                 | Expiration date related to issued permit.       |
|    26 | 19.138 | Estimated Cost                         | Initial estimation of the cost of the project   |
|    27 |  3.049 | Revised Cost                           | Revised estimation of the cost of the project   |
|    28 | 20.670 | Existing Use                           | Existing use of the building                    |
|    29 | 25.911 | Existing Units                         | Existing number of units                        |
|    30 | 21.336 | Proposed Use                           | Proposed use of the building                    |
|    31 | 25.596 | Proposed Units                         | Proposed number of units                        |
|    32 | 18.757 | Plansets        | Plan representation indicating the general design intent of the foundation..            |
|    33 | 99.998 | TIDF Compliance                        | TIDF compliant or not, this is a new legal requirement           |
|    34 | 21.802 | Existing Construction Type         | Construction type, existing,as categories represented numerically    |
|    35 | 21.802 | Existing Construction Type Description | Descr. of the above, eg.: wood or other construction types       |
|    36 | 21.700 | Proposed Construction Type         | Construction type, proposed, as categories represented numerically   |
|    37 | 21.700 | Proposed Construction Type Description | Description of the above                                         |
|    38 | 97.305 | Site Permit                            | Permit for site                                                  |
|    39 |  0.863 | Supervisor District                    | Supervisor District to which the building location belongs to    |
|    40 |  0.867 | Neighborhoods - Analysis Boundaries    | Neighborhood to which the building location belongs to           |
|    41 |  0.862 | **Zipcode**                            | Zipcode of building address                                      |
|    42 |  0.854 | Location                               | Location in latitude, longitude pair.                            |
|    43 |      0 | Record ID                              | Some ID, not useful for this                                     |

## Understand this code to perform the group imputation:

```python
# First off, load the data
df = pd.read_csv("titanic/train.csv", index_col='PassengerId')

# We seem to be selecting 3 coloumns to use...
group_cols = ['Sex','Pclass','Title']
# This finds the average age of people who are in those specific groups
# E.g. The average age of all (Female, Lower-Class, Mrs)
impute_map = df.groupby(group_cols).Age.mean().reset_index(drop=False)

for index, row in impute_map.iterrows(): # Iterate all group possibilities
    # A Boolean column, true when the sample meets our criteria (In these specific groups)
    ind = (df[group_cols] == row[group_cols]).all(axis=1) # Returns Boolean column with the length of dataframe        
    # Replace any missing values with the average age of matching data
    df[ind] = df[ind].fillna(row["Age"])
```
This is a good example of how to make filling with means more effective, as apposed to just "The average age of all passengers"
<br>By narrowing down into specific groups, we get a better guess for that person's age, and don't distort our data as much!


## Optional External Exercises:

From Kaggle [data cleaning mini course](https://www.kaggle.com/learn/data-cleaning) do:
- [Handling Missing Values](https://www.kaggle.com/alexisbcook/handling-missing-values) Data Cleaning: 1 of 5
- [Inconsistent Data Entry](https://www.kaggle.com/alexisbcook/inconsistent-data-entry) Data Cleaning: 5 of 5
