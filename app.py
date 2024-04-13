import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('Lab19_Dataset.csv')

# Extract features (X) and targets (y) from the dataset
X = pd.get_dummies(df[['SV_01', 'SV_02', 'SV_03', 'SV_04', 'SV_05', 'SV_06', 'SV_07', 'SV_08', 'SV_09', 'Flow SP (GPM)']])
y = df[['Flow Rate (GPM)', 'C_Valve %Open', 'DPT_01 (PSI)', 'DPT_02 (PSI)', 'DPT_03 (PSI)', 'DPT_04 (PSI)',
        'DPT_05 (PSI)', 'DPT_06 (PSI)', 'DPT_07 (PSI)', 'DPT_08 (PSI)', 'DPT_09 (PSI)', 'DPT_10 (PSI)',
        'DPT_11 (PSI)', 'GPT_01 (PSI)', 'GPT_02 (PSI)', 'GPT_03 (PSI)']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and calculate mean error
y_train_pred = rf_model.predict(X_train)
mean_error = np.sqrt(mean_squared_error(y_train, y_train_pred))

def determine_loop(sv_values):
    if sv_values == [0, 0, 0, 1, 0, 0, 1, 1, 0]:
        return "Venturi Loop"
    elif sv_values == [0, 0, 0, 0, 1, 0, 1, 1, 0]:
        return "Gate Loop"
    elif sv_values == [0, 0, 0, 0, 0, 1, 1, 1, 0]:
        return "Globe Loop"
    elif sv_values == [0, 0, 0, 0, 0, 0, 1, 1, 0]:
        return "Galvanized Loop"
    elif sv_values == [1, 0, 0, 0, 0, 0, 1, 1, 0]:
        return "SCH80 Loop"
    elif sv_values == [0, 1, 0, 0, 0, 0, 1, 1, 0]:
        return "SCH40 Loop"
    elif sv_values == [1, 1, 0, 0, 0, 0, 1, 1, 0]:
        return "SCH40/SCH80 Loop"
    else:
        return "Unknown Loop"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sv_config_str = data['sv_config_str']
        flow_sp = data['flow_sp']

        sv_values = [int(char) for char in sv_config_str]
        sv_input = pd.get_dummies(pd.DataFrame([sv_values + [flow_sp]], columns=X.columns)).reindex(columns=X.columns, fill_value=0)

        # Make prediction
        prediction = rf_model.predict(sv_input)

        # Apply perturbation
        perturbation = random.uniform(-0.05, 0.05)
        perturbed_prediction = [round(value + value * perturbation, 2) for value in prediction[0]]

        # Determine the loop based on SV values
        loop = determine_loop(sv_values)

        response = {
            'flow_rate': perturbed_prediction[0],
            'c_valve_percent_open': perturbed_prediction[1],
            'dpt_01': perturbed_prediction[2],
            'dpt_02': perturbed_prediction[3],
            'dpt_03': perturbed_prediction[4],
            'dpt_04': perturbed_prediction[5],
            'dpt_05': perturbed_prediction[6],
            'dpt_06': perturbed_prediction[7],
            'dpt_07': perturbed_prediction[8],
            'dpt_08': perturbed_prediction[9],
            'dpt_09': perturbed_prediction[10],
            'dpt_10': perturbed_prediction[11],
            'dpt_11': perturbed_prediction[12],
            'gpt_01': perturbed_prediction[13],
            'gpt_02': perturbed_prediction[14],
            'gpt_03': perturbed_prediction[15],
            'mean_error': round(mean_error, 2),
            'loop': loop
        }
        return jsonify(response)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'})

if __name__ == '__main__':
    app.run(debug=True, port=9000)
