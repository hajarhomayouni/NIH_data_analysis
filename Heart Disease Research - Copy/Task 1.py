#Importing pandas. Used for reading in csv files
import pandas as pd
import numpy as np

desired_width = 320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns', 52)

#Reading in Heart_Disease.csv file which contains all other collected data on Heart Disease publications.
data1 = pd.read_csv("Heart_disease.csv")

#Reading in cwurData.csv file which contains the university world rankings we want to add as new column
data2 = pd.read_csv("cwurData.csv")

#Dropping duplicate rows by using the institution and country columns as identifiers.
#keep='last' Makes it so that the last or most recent duplicate entry is kept while the others are omitted.
newdf = data2.drop_duplicates(subset=['Institution', 'country'], keep='last').reset_index(drop=True)

#Joins the two csv files.
df1 = pd.merge(data1, newdf[['Institution', 'world_rank']], on='Institution', how='left')

#Instead of saving to the existing Heart_Diseae.csv this will create a new file. This is a precaution in case the original csv file is needed.
df1.to_csv('Updated_Heart_Disease.csv')

#print(df1)

