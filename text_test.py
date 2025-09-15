import requests

API = "http://127.0.0.1:8000"

payload = {
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Cumhuriyetin ilanının sonuçlarını açıkla"}]}
    ]
}

r = requests.post(f"{API}/kpss-text", json=payload, timeout=60)
print("Status:", r.status_code)
print(r.text)