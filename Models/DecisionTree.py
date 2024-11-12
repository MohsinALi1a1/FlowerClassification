import pickle
import numpy as np
import pandas as pd
import pandas as pd


columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']

# Load the data
df = pd.read_csv('../Dataset/iris.data', names=columns)
# Check for missing values in each column
missing_values = df.isnull().sum()
print(missing_values)

df.head()
# Some basic statistical analysis about the data
df.describe()




# Seperate features and target
data = df.values
X = data[:,0:4]
Y = data[:,4]
# Decision Tree algorithm
from sklearn.tree import DecisionTreeClassifier

# Split the data to train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Instantiate Decision Tree classifier
dt_classifier = DecisionTreeClassifier()

# Train the Decision Tree classifier
dt_classifier.fit(X_train, y_train)

# Predict from the test dataset
predictions = dt_classifier.predict(X_test)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
print("Accuracy of Decison Tree", accuracy_score(y_test, predictions))

# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

#Prediction of the species from the input vector
X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
prediction = dt_classifier.predict(X_new)
print("Prediction of Species with Desicion Tree: {}".format(prediction))

with open('../DECISIONTree.pickle', 'wb') as f:
    pickle.dump(dt_classifier, f)
