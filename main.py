from flask import Flask, jsonify, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


app = Flask(__name__)

# Load the dataset
df = pd.read_csv('Lab19_Dataset.csv')
X = df[['Flow SP (GPM)']]
y = df[['Flow Rate (GPM)', 'C_Valve %Open', 'DPT_01 (PSI)', 'DPT_02 (PSI)', 'DPT_03 (PSI)']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    flow_sp = data['flow_sp']

    # Make predictions
    prediction = rf_model.predict([[flow_sp]])

    # Prepare the response
    response = {
        'predicted_flow_rate': prediction[0][0],
        'predicted_c_valve_percent_open': prediction[0][1],
        'predicted_dpt_01': prediction[0][2],
        'predicted_dpt_02': prediction[0][3],
        'predicted_dpt_03': prediction[0][4],
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=8000)




