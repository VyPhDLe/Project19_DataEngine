import requests

url = "http://127.0.0.1:5000/predict"


data = {"flow_sp": 30.0}
response = requests.post(url, json=data)

print(response.json())