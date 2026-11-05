"""Send one ECG from a sample CSV to a running API and print the prediction.

Usage: python scripts/predict_sample.py [samples/sample_upload_TEST_PATIENT.csv] [http://127.0.0.1:8000]
"""
import sys

import pandas as pd
import requests

csv = sys.argv[1] if len(sys.argv) > 1 else "samples/sample_upload_TEST_PATIENT.csv"
url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"

df = pd.read_csv(csv)
signal = df[[c for c in df.columns if c.startswith("channel-")]].values.tolist()
resp = requests.post(f"{url}/predict", json={"signal": signal}, timeout=60)
print(resp.status_code)
body = resp.json()
print({k: body.get(k) for k in ("prediction", "confidence", "probabilities", "explanation")})
