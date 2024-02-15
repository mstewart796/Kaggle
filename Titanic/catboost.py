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

# Split into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)

# Training the CatBoost model on the Training set
from catboost import CatBoostClassifier
# Flatten the 'y' array using ravel()
y_flattened = y_train.ravel()

# Create and train the CatBoost Classifier
classifier = CatBoostClassifier()
classifier.fit(X_train, y_flattened)

# Checking accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_valid)
cm = confusion_matrix(y_valid, y_pred)
print(cm)
print(accuracy_score(y_valid, y_pred))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_flattened, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Applying Grid Search to find the best model and the best parameters

from catboost import Pool, cv

# Define the parameter grid
params = {'depth': [4, 6, 8],
          'learning_rate': [0.03, 0.1],
          'l2_leaf_reg': [1, 3, 5, 7, 9]}

# Define the CatBoostClassifier
clf = CatBoostClassifier(iterations=1000)

# Perform grid search
grid = clf.grid_search(params, X=Pool(X_train, label=y_train), cv=3)

# Get best parameters
best_params = grid['params']

print("Best parameters:", best_params)


# Applying best parameters
classifier = CatBoostClassifier(depth=8, l2_leaf_reg=1, learning_rate=0.03)
classifier.fit(X_train, y_flattened)

# Checking accuracy again
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_flattened, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))