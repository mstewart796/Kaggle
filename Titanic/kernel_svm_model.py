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

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
# Flatten the 'y' array using ravel()
y_flattened = y_train.ravel()

# Create and train the SVM Classifier
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_flattened)

# Predicting the Test set results
y_pred = classifier.predict(X_valid)

# Checking accuracy
# train_accuracy = classifier.score(X, y)
# print(f'Training Accuracy: {train_accuracy}')
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_valid)
cm = confusion_matrix(y_valid, y_pred)
print(cm)
print(accuracy_score(y_valid, y_pred))