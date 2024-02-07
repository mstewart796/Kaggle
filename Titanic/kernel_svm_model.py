# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('titanic_train.csv')
X = dataset.iloc[:, [2, 4, 5]].values
y = dataset.iloc[:, 1].values

df_test = pd.read_csv('titanic_test.csv')
X_test = df_test.iloc[:, [1, 3, 4]].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, [2]])
X[:, [2]] = imputer.transform(X[:, [2]])

imputer_test = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_test.fit(X_test[:, [2]])
X_test[:, [2]] = imputer.transform(X_test[:, [2]])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X_test[:, 1] = le.fit_transform(X_test[:, 1])

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
# Flatten the 'y' array using ravel()
y_flattened = y.ravel()

# Create and train the DecisionTreeClassifier
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, y_flattened)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Checking accuracy
train_accuracy = classifier.score(X, y)
print(f'Training Accuracy: {train_accuracy}')