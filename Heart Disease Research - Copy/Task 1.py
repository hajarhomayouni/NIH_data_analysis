#Importing pandas. Used for reading in csv files
import pandas as pd
import numpy as np

desired_width=320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns',52)

#Reading in Heart_Disease.csv file which contains all other collected data on Heart Disease publications.
data1 = pd.read_csv("Heart_disease.csv")
#data1 = pd.read_csv("Heart_disease.csv", sep='\t')
#data1.columns = data1.columns.str.replace(' ', '')
#HeartDisease


data2 = pd.read_csv("cwurData.csv")
newdf = data2.drop_duplicates(subset=['Institution', 'country'], keep='last').reset_index(drop=True)
#data2.columns = data2.columns.str.replace(' ', '')

#Joins the two csv files.
#data1.join(data2.set_index('Institution'), on='Institution') #FIXME combines both csv files and adds on unneccessary columns from "cswurData.csv"
#pd.set_option('display.max_columns', 52)
df1 = pd.merge(data1, newdf[['Institution', 'world_rank', 'year']], on='Institution', how='left')

#df1 = df1.astype('str')
#df1 = df1.drop_duplicates(subset=None, keep='last', inplace=False)
#df2 = df1.drop_duplicates(subset=None, keep='last', inplace=False)
print(df1)
#print(data1.join(data2.set_index('Institution'), on='Institution'))
