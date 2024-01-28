from flask import Flask, jsonify, request
import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import time

app = Flask(__name__)

df = pd.read_csv('Lab19_Dataset.csv')
loop_names = df['Loop'].unique()

loop_models = {}
loop_mean_errors = {}

for loop_name in loop_names:
    loop_data = df[df['Loop'] == loop_name]

    X = loop_data[['Flow SP (GPM)']]
    y = loop_data[['Flow Rate (GPM)', 'C_Valve %Open', 'DPT_01 (PSI)', 'DPT_02 (PSI)', 'DPT_03 (PSI)', 'DPT_04 (PSI)',
                                                       'DPT_05 (PSI)', 'DPT_06 (PSI)', 'DPT_07 (PSI)', 'DPT_08 (PSI)',
                                                       'DPT_09 (PSI)', 'DPT_10 (PSI)', 'DPT_11 (PSI)', 'GPT_01 (PSI)',
                                                       'GPT_02 (PSI)', 'GPT_03 (PSI)']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_model.fit(X_train, y_train)

    loop_models[loop_name] = rf_model

    y_train_pred = rf_model.predict(X_train)

    mean_error = np.sqrt(mean_squared_error(y_train, y_train_pred))
    loop_mean_errors[loop_name] = mean_error


@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()

        data = request.json
        loop_name = data['loop_name']
        flow_sp = data['flow_sp']

        if loop_name not in loop_models:
            raise ValueError("Invalid 'loop_name'")

        rf_model = loop_models[loop_name]
        prediction = rf_model.predict([[flow_sp]])

        perturbation = random.uniform(-0.05, 0.05)
        perturbed_prediction = [value + value * perturbation for value in prediction[0]]

        elapsed_time = time.time() - start_time

        response = {
            'loop_name': loop_name,
            'predicted_flow_rate': perturbed_prediction[0],
            'predicted_c_valve_percent_open': perturbed_prediction[1],
            'predicted_dpt_01': perturbed_prediction[2],
            'predicted_dpt_02': perturbed_prediction[3],
            'predicted_dpt_03': perturbed_prediction[4],
            'predicted_dpt_04': perturbed_prediction[5],
            'predicted_dpt_05': perturbed_prediction[6],
            'predicted_dpt_06': perturbed_prediction[7],
            'predicted_dpt_07': perturbed_prediction[8],
            'predicted_dpt_08': perturbed_prediction[9],
            'predicted_dpt_09': perturbed_prediction[10],
            'predicted_dpt_10': perturbed_prediction[11],
            'predicted_dpt_11': perturbed_prediction[12],
            'predicted_gpt_01': perturbed_prediction[13],
            'predicted_gpt_02': perturbed_prediction[14],
            'predicted_gpt_03': perturbed_prediction[15],
            'mean_error': loop_mean_errors[loop_name],
            'elapsed_time': elapsed_time
        }

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run(debug=True)
