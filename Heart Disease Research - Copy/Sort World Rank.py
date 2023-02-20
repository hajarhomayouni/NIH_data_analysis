#Importing pandas. Used for reading in csv files
import pandas as pd
import numpy as np
desired_width=320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns',52)
pd.set_option('display.max_rows', 1000)

data2 = pd.read_csv("cwurData.csv")

# drop rows which have same order_id
# and customer_id and keep latest entry
newdf = data2.drop_duplicates(subset=['Institution', 'country'], keep='last').reset_index(drop=True)

# print latest dataframe
print(newdf)