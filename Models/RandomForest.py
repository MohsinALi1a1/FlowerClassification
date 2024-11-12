import numpy as np
import pandas as pd
import pickle

columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']

# Load the data
df = pd.read_csv('../Dataset/iris.data', names=columns)
print(df.head())
print(df.describe())
data = df.values
X = data[:, 0:4]
Y = data[:, 4]

# New input data for prediction
X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])

# Import RandomForestClassifier from sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Instantiate the RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier
random_forest.fit(X_train, y_train)

# Predict using the test dataset
predictions = random_forest.predict(X_test)

# Calculate accuracy
print("Accuracy of Random Forest Classifier:", accuracy_score(y_test, predictions))

# Print a detailed classification report
print(classification_report(y_test, predictions))

# Prediction of species from the input vector
prediction = random_forest.predict(X_new)
print("Prediction of Species with Random Forest: {}".format(prediction))

# Save the trained model as a pickle file
with open('../random_forest_model.pickle', 'wb') as f:
    pickle.dump(random_forest, f)
