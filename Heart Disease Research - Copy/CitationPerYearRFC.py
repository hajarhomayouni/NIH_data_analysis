import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix


# Reading in Heart_Disease.csv file which contains all other collected data on Heart Disease publications.
data1 = pd.read_csv("Updated_Heart_Disease.csv")

# Predict citation_per_year based on world_rank, journal_ranking, year, and gender.

# dividing the dataset into training and testing datasets
X = data1[['Year', 'Rank', 'world_rank']]
Y = data1[['Citations Per Year']]

# splitting in random train/test subsets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

classifier = RandomForestClassifier(n_estimators=100)  # Create random forest classifier

classifier.fit(X_train, Y_train)

Y_prediction = classifier.predict(X_test)

print(confusion_matrix(Y_test, Y_prediction))
print()
print("Model Accuracy: ", metrics.accuracy_score(Y_test, Y_prediction))
print()
print(classification_report(Y_test,Y_prediction))

