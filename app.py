import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='WebPage')


with open('SVM.pickle', 'rb') as f:
    svm_model = pickle.load(f)
with open('DECISIONTree.pickle', 'rb') as f:
    dt_model = pickle.load(f)
with open('logistic_regression.pickle', 'rb') as f:
    lr_model = pickle.load(f)
with open('random_forest_model.pickle', 'rb') as f:
    rf_model = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/Result', methods=['POST'])
def result():
    try:
        feature1 = float(request.form.get('feature1'))
        feature2 = float(request.form.get('feature2'))
        feature3 = float(request.form.get('feature3'))
        feature4 = float(request.form.get('feature4'))

        # Prepare input for prediction
        X_new = np.array([[feature1, feature2, feature3, feature4]])

        # Make predictions using Logistic Regression (or choose the best model)
        lr_prediction = lr_model.predict(X_new)




        processed_data = f"Predicted species for features {X_new.flatten()} is {lr_prediction}"
    except ValueError:
        return render_template('index.html', error='Invalid input. Please enter valid numbers.')

    return render_template('index.html', processed_data=processed_data)


if __name__ == '__main__':
    app.run(debug=True)
