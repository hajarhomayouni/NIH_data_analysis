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
data1['Topic'] = data1['Topic'].astype('category')

# This will assign numerical values that will be stored in another column.
# The values are stored in the gender_new and topic_ new columns.
data1['gender_new'] = data1['ga_gender'].cat.codes
data1['topic_new'] = data1['Topic'].cat.codes

# New instance of one hot encoder
prep = OneHotEncoder()

#
encoded_gender = pd.DataFrame(prep.fit_transform(data1[['gender_new']]).toarray())
encoded_topic = pd.DataFrame(prep.fit_transform(data1[['topic_new']]).toarray())

encoded_data1 = data1.join(encoded_gender)
encoded_data1 = data1.join(encoded_topic)

# print(encoded_data1)

from sklearn.preprocessing import MinMaxScaler

df_sklearn = encoded_data1.copy()

column1 = 'Year'
column2 = 'Rank'
column3 = 'Citations Per Year'

# Using MinMaxScaler to normalize each column
df_sklearn[column1] = MinMaxScaler().fit_transform(np.array(df_sklearn[column1]).reshape(-1, 1))
# print(df_sklearn[column1].value_counts()[1.0])
# print(df_sklearn[column1].value_counts()[0.0])
#print(df_sklearn[column1].value_counts()[-1.0])
df_sklearn['APT'] = MinMaxScaler().fit_transform((np.array(df_sklearn['APT']).reshape((-1, 1))))
df_sklearn[column2] = MinMaxScaler().fit_transform(np.array(df_sklearn[column2]).reshape(-1, 1))
df_sklearn[column3] = MinMaxScaler().fit_transform(np.array(df_sklearn[column3]).reshape(-1, 1))

# Predict citation_per_year based on world_rank, journal_ranking, year, and gender.
# Need to add gender back to this now that I know labelEncoder works.
# dividing the dataset into training and testing datasets
# Leaving out world_rank for the time being, look for larger data sets
# Including APT for testing purposes. See if it has any impact on the final outcome.

X = df_sklearn[['Year', 'APT', 'Rank', 'gender_new', 'topic_new']]
# print(X)
y = encoded_data1['Citations Per Year']

# print(df_sklearn)
#
transform = preprocessing.LabelEncoder()
#
y_transformed = transform.fit_transform(y)


# Y or predicted data column must be categorical or in other words 0 or 1
# splitting in random train/test subsets
#
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.30)
#
classifier = RandomForestClassifier(max_depth=32, n_estimators=100, min_samples_split=2, max_leaf_nodes=6)
# Create random forest classifier
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
print()
# print(classifier.predict([[1, 0.33, 0]]))

#use APT
