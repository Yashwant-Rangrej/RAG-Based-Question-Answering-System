import requests
import json
import time

URL = "http://localhost:8000"
HEADERS = {"X-API-Key": "dev-secret-key-12345", "Content-Type": "application/json"}

def test_hybrid(query):
    print(f"\nTesting Query: {query}")
    payload = {"query": query, "top_k": 5}
    try:
        response = requests.post(f"{URL}/ask", headers=HEADERS, json=payload, stream=True, timeout=120)
        print(f"Status: {response.status_code}")
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                # print(f"DEBUG RAW: {decoded_line}")
                if decoded_line.startswith('data: '):
                    json_str = decoded_line.replace('data: ', '').strip()
                    if json_str == '[DONE]':
                        print("\n[DONE]")
                        break
                    try:
                        data = json.loads(json_str)
                        if 'token' in data:
                            print(data['token'], end="", flush=True)
                        if 'sources' in data and data['sources']:
                            print(f"\n(Sources: {len(data['sources'])} chunks)")
                        if 'answer' in data: # Handle non-streaming fallback
                             print(f"Fallback Answer: {data['answer']}")
                    except Exception as e:
                        print(f"\n[Parse Error: {e} | Raw: {json_str}]")
    except Exception as e:
        print(f"Request failed: {e}")
    print("\n" + "="*50)

if __name__ == "__main__":
    test_hybrid("What is the secret code?")
    test_hybrid("What is the capital of France?")
