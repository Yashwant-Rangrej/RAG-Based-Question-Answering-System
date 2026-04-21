import requests
import json

URL = "http://localhost:8000"
HEADERS = {"X-API-Key": "dev-secret-key-12345", "Content-Type": "application/json"}

def test_ask():
    payload = {"query": "What is the secret code?", "top_k": 5}
    response = requests.post(f"{URL}/ask", headers=HEADERS, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    test_ask()
