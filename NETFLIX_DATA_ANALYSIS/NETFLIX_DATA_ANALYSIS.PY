# Netflix is known for its work in data science, AI,
# and ML, particularly for building strong recommendation models
# and algorithms that understand customer behavior and patterns. 
# Suppose you are working in a data-driven job role, and you have 
# a dataset of more than 9,000 movies. You need to solve the following
# questions to help the company make informed business decisions 
# accordingly.

# 1 What is the most frequent genre of movies released on Netflix?

# 2 What genres has highest votes?

# 3 What movie got the highest popularity? what's its genre?

# 4 What movie got the lowest popularity? what's its genre?

# 5 Which year has the most filmed movies?



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("mymoviedb.csv",engine="python", on_bad_lines="skip", encoding="latin1")
print(df)

#data information 
print(df.info())


#onliy goners 
ganer = df["Genre"]
print(ganer.head(10))

# Find the duplicate valu 
print(df.duplicated().sum())