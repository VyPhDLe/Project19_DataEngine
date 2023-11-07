import requests

url = "http://127.0.0.1:8000/predict"


data = {"flow_sp": 28.0}
response = requests.post(url, json=data)

print(response.json())