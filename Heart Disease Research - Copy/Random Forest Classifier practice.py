import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# Predict citation_per_year based on world_rank, journal_ranking, year, and gender.

iris = datasets.load_iris()

# dividing the dataset into training and testing datasets
X, Y = datasets.load_iris(return_X_y= True)

# splitting in random train/test subsets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)


# New iris dataframe
df = pd.DataFrame({'sepalwidth': iris.data[:, 0], 'sepallength': iris.data[:, 1],
                   'petalwidth': iris.data[:, 2], 'petallength': iris.data[:, 3], 'species': iris.target})

classifier = RandomForestClassifier(n_estimators=100)  # Create random forest classifier

classifier.fit(X_train, Y_train)

Y_prediction = classifier.predict(X_test)

print()

print("Model Accuracy: ", metrics.accuracy_score(Y_test, Y_prediction))

print(confusion_matrix(Y_test, Y_prediction))
print()
print(classification_report(Y_test, Y_prediction))