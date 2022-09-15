import pickle
import json
from flask import Flask, request, app, jsonify, render_template, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
my_model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1,-1)
    print(final_input)
    output=my_model.predict(final_input)[0]
    return render_template("index.html", prediction_text="The probability of admission is {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)