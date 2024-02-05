import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Import the dataset
df = pd.read_csv('Iris.csv')
# df.head()

# Drop the id column and check for null values
df.drop('Id', axis=1, inplace=True)
# df.columns
df.info()
# df.isnull().sum()

# Set X and y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Train the SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Check accuracy
print(classifier.score(X_train, y_train))
# 98.2%

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
# 97.4%
