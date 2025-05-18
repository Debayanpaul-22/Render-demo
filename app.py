import pickle

import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import *

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route("/")
def home():
    return "Welcome to UPI Detection System"


@app.route('/predict', methods=['POST'])
def predict():
    Transaction_Type = request.form.get('Transaction_Type')
    Payment_Gateway = request.form.get('Payment_Gateway')
    Transaction_City = request.form.get('Transaction_City')
    Transaction_State = request.form.get('Transaction_State')
    Transaction_Status = request.form.get('Transaction_Status')
    Device_OS = request.form.get('Device_OS')
    Transaction_Frequency = request.form.get('Transaction_Frequency')
    Merchant_Category = request.form.get('Merchant_Category')
    Transaction_Channel = request.form.get('Transaction_Channel')
    Transaction_Amount_Deviation = request.form.get('Transaction_Amount_Deviation')
    Days_Since_Last_Transaction = request.form.get('Days_Since_Last_Transaction')
    amount = request.form.get('amount')

    input_quary = np.array([[Transaction_Type, Payment_Gateway, Transaction_City, Transaction_State, Transaction_Status
                                , Device_OS, Transaction_Frequency, Merchant_Category, Transaction_Channel
                                , Transaction_Amount_Deviation, Days_Since_Last_Transaction, amount]])

    label_encoders = {}
    new_df = pd.DataFrame(input_quary)
    for column in new_df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        new_df[column] = le.fit_transform(new_df[column].astype(str))
        label_encoders[column] = le

    result = model.predict(new_df)[0]

    return jsonify({'fraud': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
