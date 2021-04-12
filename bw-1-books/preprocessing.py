import numpy as np
import pandas as pd
import re

"""
This is the pre-processing for Build Week 1 of Strive (Goodreads Book Data)
The preprocessing consists of:
Taking a csv file from the scrape() function and putting it into a pandas dataframe
Transforming the awards from a list of awards to a number
Normalise the average ratings and put them inside the value range 1-10

+ I've tidied up some of the other columns in case we use them later on

"""


def mean_normalise(ratings_series):
    print(">> Normalising {} with repsect to the mean".format(ratings_series.name))
    # Normalise the passed data
    normalised = ((ratings_series - np.mean(ratings_series)) / (max(ratings_series) - min(ratings_series)))
    # Move into the range of 1-10
    print(">> Transforming values to between 1 and 10")
    one_to_ten = ((normalised + 1) * 4.5) + 1
    return one_to_ten


def clean_num_pages(num_pages_series):
    print(">> Cleaning {}".format(num_pages_series.name))
    # For num_pages, we only want the number of pages so we remove the excess text (e.g. "100 pages" --becomes-> 100)
    return (num_pages_series.str.replace(" pages", "")).astype(int)


def clean_publish_year(original_publish_year_series):
    print(">> Cleaning {}".format(original_publish_year_series.name))
    # For original_publish year, we only need the year, which we can find via a RegEx that looks for 4 sequential numbers (e.g. 1984)
    return original_publish_year_series.str.extract(r'([0-9]{4})', expand=False).astype(int)


def clean_series(book_series_series):
    print(">> Cleaning {}".format(book_series_series.name))
    # For series, we only need to see if something IS a series or not, so we can use a boolean
    return book_series_series.notnull()


def clean_awards(awards_series):
    print(">> Cleaning {}".format(awards_series.name))
    # For Awards, we only want the number of awards, We'll count how many (YEAR)s there are, and replace any NaNs with a 0
    return (awards_series.str.count(r"([0-9]{4})")).replace(np.nan, 0)


def clean_genres(genres_series):
    print(">> Cleaning {}".format(genres_series.name))
    # For genres, I'm just going to replace "Science Fiction" with "Sci-Fi"
    return genres_series.str.replace("Science Fiction", "Sci-Fi")


def clean_data(dataframe):
    df = dataframe
    print("-- Cleaning data...")
    df.num_pages = clean_num_pages(df.num_pages)
    df.original_publish_year = clean_publish_year(df.original_publish_year)
    df.series = clean_series(df.series)
    df.awards = clean_awards(df.awards)
    df.genres = clean_genres(df.genres)
    print("-- Cleaning complete!")


def convert_ratings(dataframe):
    df = dataframe
    print("Converting ratings...")
    df["mean_norm_ratings"] = mean_normalise(df.avg_rating)
    print("Conversion complete!")


def locate_data(csv_path):
    print("Searching for file: {}".format(csv_path))

    return pd.read_csv(csv_path)


def preprocessing(csv_path):
    print("Starting Preprocessing with [{}]...".format(csv_path))
    df = pd.read_csv(csv_path)
    clean_data(df)
    convert_ratings(df)
    print("Preprocessing complete!")
    return df


if __name__ == "__main__":
    csv_file_path = "books.csv"
    print("CSV file: {}".format(csv_file_path))
    preprocessing(csv_file_path)

