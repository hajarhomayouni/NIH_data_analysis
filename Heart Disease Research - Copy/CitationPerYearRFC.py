import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn import utils

desired_width = 320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns', 52)


# Reading in Heart_Disease.csv file which contains all other collected data on Heart Disease publications.
data1 = pd.read_csv("Updated_Heart_Disease.csv")

# Replacing NULL values
data1["world_rank"].fillna(0, inplace=True)

data1['ga_gender'].fillna('unknown', inplace=True)

# Converting data type to category/categorical.
# Same process will be used for topic column which will exist later on.
data1['ga_gender'] = data1['ga_gender'].astype('category')

# This will assign numerical values that will be stored in another column.
# The values are stored in the gender_new column.
data1['gender_new'] = data1['ga_gender'].cat.codes

# New instance of one hot encoder
prep = OneHotEncoder()

#
encoded_gender = pd.DataFrame(prep.fit_transform(data1[['gender_new']]).toarray())

encoded_data1 = data1.join(encoded_gender)

# print(encoded_data1)

from sklearn.preprocessing import MinMaxScaler

df_sklearn = encoded_data1.copy()

column1 = 'Year'
column2 = 'Rank'
column3 = 'Citations Per Year'

# Using MinMaxScaler to normalize each column
df_sklearn[column1] = MinMaxScaler().fit_transform(np.array(df_sklearn[column1]).reshape(-1, 1))
df_sklearn[column2] = MinMaxScaler().fit_transform(np.array(df_sklearn[column2]).reshape(-1, 1))
df_sklearn[column3] = MinMaxScaler().fit_transform(np.array(df_sklearn[column3]).reshape(-1, 1))

#print(df_sklearn)



# year_arr = np.array(data1['Year'])
# rank_arr = np.array(data1['Rank'])
#
# normalized_year = preprocessing.normalize([year_arr])
# normalized_rank = preprocessing.normalize([rank_arr])
#
# print("Normalized rank: ", normalized_rank)
# print("Normalized year: ", normalized_year)

# year_df = pd.DataFrame(normalized_year, columns=['Normalized_year'])
# year_df.reset_index(drop=True, inplace=True)
# print(year_df)
# rank_df = pd.DataFrame(normalized_rank, columns=['Normalized_rank'])
# print(rank_df)
# rank_df.reset_index(drop=True, inplace=True)

# encoded_data1.reset_index(drop=True, inplace=True)

# Preprocessed_data = pd.concat([encoded_data1, year_df, rank_df], axis=1)

# encoded_data1 = data1.join(year_df)
# encoded_data1 = data1.join(rank_df)

#print(Preprocessed_data)

# print(X)

# Predict citation_per_year based on world_rank, journal_ranking, year, and gender.
# Need to add gender back to this now that I know labelEncoder works.
# dividing the dataset into training and testing datasets
# Leaving out world_rank for the time being, look for larger data sets

X = df_sklearn[['Year', 'Rank', 'gender_new']]
y = encoded_data1['Citations Per Year']

#print(df_sklearn)

transform = preprocessing.LabelEncoder()

y_transformed = transform.fit_transform(y)
# Y or predicted data column must be categorical or in other words 0 or 1



# splitting in random train/test subsets
#
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.30)
#
classifier = RandomForestClassifier(max_depth=5, n_estimators=100)  # Create random forest classifier
#
classifier.fit(X_train, y_train)
#
y_prediction = classifier.predict(X_test)
#
print(confusion_matrix(y_test, y_prediction))
print()
print("Model Accuracy: ", metrics.accuracy_score(np.array(y_test), y_prediction))
print(metrics.f1_score(y_test, y_prediction, average='weighted', labels=np.unique(y_prediction)))
print(classification_report(y_test, y_prediction, zero_division=0))

