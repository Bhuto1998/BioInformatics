# In this file we will split the total data into 80% train and 20% test and we will work with the train data for the time being
# i.e. till the testing stage

# imports
import numpy as np
import pandas as pd
# Reading the data
input_file = "data_remastered_final.csv"
df = pd.read_csv(input_file)
# Shuffling the dataset
df = df.reindex(np.random.permutation(df.index))
# Splitting the Dataset
size = len(df)
size = int (size*0.8)
df_train = df[:size]
df_test = df[size:]

# Saving the test and train data
name = "train.csv"
df_train.to_csv(name)
name = "test.csv"
df_test.to_csv(name)

