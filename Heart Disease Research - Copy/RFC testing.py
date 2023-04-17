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
data1 = pd.read_csv("RFCcolumns.csv")

# Replacing NULL values
# data1["world_rank"].fillna(0, inplace=True)
#
# data1['ga_gender'].fillna('unknown', inplace=True)

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

# X = encoded_data1[['Year', 'APT', 'Rank', 'gender_new']]

X = encoded_data1[['Year', 'APT', 'Rank', 'gender_new']]
# X.columns = X.columns.astype(str)
y = encoded_data1['Citations Per Year']
print(X)

# View count of each class
print (y.value_counts())


# print(df_sklearn)
#
transform = preprocessing.LabelEncoder()
#
y_transformed = transform.fit_transform(y)


# Y or predicted data column must be categorical or in other words 0 or 1
# splitting in random train/test subsets
#
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, random_state=1, test_size=0.30)
# print (y.value_counts())
#
classifier = RandomForestClassifier(max_depth=32, n_estimators=10, min_samples_split=2, max_features="sqrt", max_leaf_nodes=6)
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