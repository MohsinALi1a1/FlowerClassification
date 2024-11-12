import pickle
import numpy as np
import pandas as pd


columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']

# Load the data
df = pd.read_csv('../Dataset/iris.data', names=columns)


df.head()

# Some basic statistical analysis about the data
df.describe()

# Seperate features and target
data = df.values
X = data[:,0:4]
Y = data[:,4]
X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])

# Logistic Regression algorithm
from sklearn.linear_model import LogisticRegression

# Split the data to train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Instantiate Logistic Regression classifier
logistic_regression = LogisticRegression()

# Train the Logistic Regression classifier
logistic_regression.fit(X_train, y_train)

# Predict from the test dataset
predictions = logistic_regression.predict(X_test)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
print("Accuracy of logistic Regression:", accuracy_score(y_test, predictions))

# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

#Prediction of the species from the input vector
prediction = logistic_regression.predict(X_new)
print("Prediction of Species with Logistic Regression: {}".format(prediction))


with open('../logistic_regression.pickle', 'wb') as f:
    pickle.dump(logistic_regression, f)