import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('homonuclear-159-24features.csv')
#pandas package has function that reads csvs and outputs in data frames - or in this case variable called data
data.head(24)
#.head outputs first five rows of dataframe

