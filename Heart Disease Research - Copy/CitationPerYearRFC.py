import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn import utils


# Reading in Heart_Disease.csv file which contains all other collected data on Heart Disease publications.
data1 = pd.read_csv("Updated_Heart_Disease.csv")

# Replacing NULL values
data1["world_rank"].fillna(0, inplace=True)
# Predict citation_per_year based on world_rank, journal_ranking, year, and gender.

# dividing the dataset into training and testing datasets
X = data1[['Year', 'Rank', 'world_rank']]
y = data1['Citations Per Year']

prep = preprocessing.LabelEncoder()
y_transformed = prep.fit_transform(y)

# splitting in random train/test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.30)

classifier = RandomForestClassifier(max_depth=5, n_estimators=1000)  # Create random forest classifier

classifier.fit(X_train, y_train)

y_prediction = classifier.predict(X_test)

print(confusion_matrix(y_test, y_prediction))
print()
print("Model Accuracy: ", metrics.accuracy_score(y_test, y_prediction))
print()
print(classification_report(y_test, y_prediction))

