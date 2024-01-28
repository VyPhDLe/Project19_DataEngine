import requests

API_URL = 'http://127.0.0.1:5000/predict'

data = {
    'loop_name': 'Venturi Loop',
    'flow_sp': 15
}

# Make a POST request to the API
response = requests.post(API_URL, json=data)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    result = response.json()
    print(f"Prediction for {result['loop_name']}:")
    print(f"Predicted Flow Rate: {result['predicted_flow_rate']}")
    print(f"Predicted C Valve Percent Open: {result['predicted_c_valve_percent_open']}")
    print(f"Predicted DPT_01: {result['predicted_dpt_01']}")
    print(f"Predicted DPT_02: {result['predicted_dpt_02']}")
    print(f"Predicted DPT_03: {result['predicted_dpt_03']}")
    print(f"Mean Error: {result['mean_error']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)