import requests

url = "http://127.0.0.1:1234/invocations"
payload = {"inputs": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}

r = requests.post(url, json=payload)
print("Status:", r.status_code)
print("Response:", r.json())