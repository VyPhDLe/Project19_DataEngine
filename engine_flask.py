import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

df = pd.read_csv('Lab19_Dataset.csv')

X = pd.get_dummies(df[['SV_01', 'SV_02', 'SV_03', 'SV_04', 'SV_05', 'SV_06', 'SV_07', 'SV_08', 'SV_09', 'Flow SP (GPM)']])
y = df[['Flow Rate (GPM)', 'C_Valve %Open', 'DPT_01 (PSI)', 'DPT_02 (PSI)', 'DPT_03 (PSI)', 'DPT_04 (PSI)',
        'DPT_05 (PSI)', 'DPT_06 (PSI)', 'DPT_07 (PSI)', 'DPT_08 (PSI)', 'DPT_09 (PSI)', 'DPT_10 (PSI)',
        'DPT_11 (PSI)', 'GPT_01 (PSI)', 'GPT_02 (PSI)', 'GPT_03 (PSI)']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)

y_train_pred = rf_model.predict(X_train)
mean_error = np.sqrt(mean_squared_error(y_train, y_train_pred))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sv_config_str = data['sv_config_str']
        flow_sp = data['flow_sp']

        sv_values = [int(char) for char in sv_config_str]
        sv_input = pd.get_dummies(pd.DataFrame([sv_values + [flow_sp]], columns=X.columns)).reindex(columns=X.columns, fill_value=0)

        prediction = rf_model.predict(sv_input)

        perturbation = random.uniform(-0.05, 0.05)
        perturbed_prediction = [value + value * perturbation for value in prediction[0]]

        response = {
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
            'mean_error': mean_error,
        }
        return jsonify(response)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'})

if __name__ == '__main__':
    app.run(debug=True)
