import requests

def send_prediction_request(sv_config_str, flow_sp):
    url = 'http://127.0.0.1:9000/predict'

    data = {
        'sv_config_str': sv_config_str,
        'flow_sp': flow_sp
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request failed with status code {response.status_code}")
        return None

if __name__ == "__main__":

    sv_config_str = "010000000"
    flow_sp = 20
    result = send_prediction_request(sv_config_str, flow_sp)

    if result is not None:
        print("Prediction Result:")
        print(result)
    else:
        print("Error occurred during prediction.")


