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

# Training the XGBoost model on the Training set
from xgboost import XGBClassifier
# Flatten the 'y' array using ravel()
y_flattened = y_train.ravel()

# Create and train the XGBoost Classifier
classifier = XGBClassifier()
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
from sklearn.model_selection import GridSearchCV
# Define parameters for grid search
parameters = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'gamma': [0, 0.1, 0.2, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Perform grid search
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)
grid_search.fit(X_train, y_flattened)

# Get best results
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


# Applying best parameters
classifier = XGBClassifier(colsample_bytree=1.0, gamma=0.3, learning_rate=0.1, max_depth=5, n_estimators=200, subsample=1.0)
classifier.fit(X_train, y_flattened)

# Checking accuracy again
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_flattened, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))