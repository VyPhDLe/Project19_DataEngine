import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import base64
import io

app = Flask(__name__)

df = pd.read_csv('Lab19_Dataset.csv')

lab_outputs = ['Flow Rate (GPM)', 'C_Valve %Open', 'DPT_01 (PSI)', 'DPT_02 (PSI)', 'DPT_03 (PSI)', 'DPT_04 (PSI)',
              'DPT_05 (PSI)', 'DPT_06 (PSI)', 'DPT_07 (PSI)', 'DPT_08 (PSI)', 'DPT_09 (PSI)', 'DPT_10 (PSI)',
              'DPT_11 (PSI)', 'GPT_01 (PSI)', 'GPT_02 (PSI)', 'GPT_03 (PSI)']
lab_inputs = ['SV_01', 'SV_02', 'SV_03', 'SV_04', 'SV_05', 'SV_06', 'SV_07', 'SV_08', 'SV_09', 'Flow SP (GPM)']

X = pd.get_dummies(df[lab_inputs])
y = df[lab_outputs]

rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(X, y)

y_pred = rf_model.predict(X)
mean_error = np.sqrt(mean_squared_error(y, y_pred))

def determine_loop(sv_values):
    if sv_values == [0, 0, 0, 1, 0, 0, 1, 1, 0]:
        return "Venturi Loop"
    elif sv_values == [0, 0, 0, 0, 1, 0, 1, 1, 0]:
        return "Gate Loop"
    elif sv_values == [0, 0, 0, 0, 0, 1, 1, 1, 0]:
        return "Globe Loop"
    elif sv_values == [1, 0, 0, 0, 0, 0, 1, 1, 0]:
        return "Galvanized Loop"
    elif sv_values == [0, 1, 0, 0, 0, 0, 1, 1, 0]:
        return "SCH80 Loop"
    elif sv_values == [0, 0, 1, 0, 0, 0, 1, 1, 0]:
        return "SCH40 Loop"
    elif sv_values == [0, 1, 1, 0, 0, 0, 1, 1, 0]:
        return "SCH40/SCH80 Loop"
    elif sv_values == [0, 0, 0, 0, 0, 0, 0, 0, 0]:
        return "Offset"
    else:
        return "Unknown Loop (*)"

@app.route('/predict', methods=['POST'])
def predict():

    global df
    try:
        data = request.json
        sv_config_str = data['sv_config_str']
        flow_sp = data['flow_sp']

        sv_values = [int(char) for char in sv_config_str]

        pump_status = 'ON' if any(sv_values) else 'OFF'

        new_row = pd.DataFrame({'Pump': pump_status,
                                'Flow SP (GPM)': flow_sp,
                                'SV_01': sv_values[0],
                                'SV_02': sv_values[1],
                                'SV_03': sv_values[2],
                                'SV_04': sv_values[3],
                                'SV_05': sv_values[4],
                                'SV_06': sv_values[5],
                                'SV_07': sv_values[6],
                                'SV_08': sv_values[7],
                                'SV_09': sv_values[8]}, index=[df.index[-1]+1])

        df = pd.concat([df, new_row])

        X = pd.get_dummies(df[lab_inputs])

        prediction = rf_model.predict(X.iloc[[-1]])

        perturbation = random.uniform(-0.05, 0.05)
        perturbed_prediction = [round(value + value * perturbation, 2) for value in prediction[0]]

        loop = determine_loop(sv_values)

        df.loc[df.index[-1], 'Flow Rate (GPM)':'GPT_03 (PSI)'] = perturbed_prediction
        df.loc[df.index[-1], 'Loop'] = loop

        sprites = {}
        for output in lab_outputs:
            plt.figure(figsize=(8, 6))

            filtered_df = df[
                df[['SV_01', 'SV_02', 'SV_03', 'SV_04', 'SV_05', 'SV_06', 'SV_07', 'SV_08', 'SV_09']].apply(
                    lambda x: list(x) == sv_values, axis=1)]
            plt.scatter(filtered_df['Flow Set Point (GPM)'], filtered_df[output], color='blue', label='Actual data recorded')
            plt.scatter(flow_sp, perturbed_prediction[lab_outputs.index(output)], color='red', label='Predicted value',
                        marker='o', s=100)

            plt.title(f'Regression for {output} (Valve Config: {sv_config_str})')
            plt.xlabel('Flow Set Point (GPM)')
            plt.ylabel(output)
            plt.legend()
            plt.savefig(f'{output}_regression.png')

            plt.close()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode('utf-8')

            sprites[output] = img_str

        df.to_csv('Lab19_Dataset.csv', index=False)

        response = {
            'pump_status': pump_status,
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
            'loop': loop,
            'sprites': sprites
        }
        return jsonify(response)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'})

if __name__ == '__main__':
    app.run(debug=True, port=9000)
