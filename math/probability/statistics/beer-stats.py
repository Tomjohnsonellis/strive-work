import pandas as pd
import numpy as np
from urllib.request import urlretrieve
URL = 'http://go.gwu.edu/engcomp2data1'
urlretrieve(URL, 'beers.csv')
beers = pd.read_csv("beers.csv")

#print(beers.columns)
# abv: Alcohol-by-volume of the beer.
# ibu: International Bittering Units of the beer.
# id: Unique identifier of the beer.
# name: Name of the beer.
# style: Style of the beer.
# brewery_id: Unique identifier of the brewery.
# ounces: Ounces of beer in the can.
#
# def exercises(series):
#     print(series[:10])
#     print(len(series))
#     series_clean = series.dropna()
#     return series_clean


# abv exercises
print("--- abv ---")
print(beers.abv[:10])
print(len(beers.abv))
abv_clean = beers.abv.dropna()
abv = abv_clean.values
print(abv)
print(len(abv))

# ibu exercises
print("--- ibu ---")
ibu_series = beers.ibu
print(ibu_series[:10])
print(len(ibu_series))
ibu_clean = ibu_series.dropna()
ibu = ibu_clean.values
print(ibu)
print(len(ibu))

# Nan-Percentage
ibu_series = beers.ibu
print(ibu_series[:10])
ibu_clean = ibu_series.dropna()
print(f"Removed {len(ibu_series) - len(ibu_clean)} NaNs!")
