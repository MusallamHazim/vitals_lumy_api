# Luminary Vitals Analysis API

MLP + LSTM neural network that analyzes a sequence of 20 vital-sign readings (HR, SpO2, RESP, TEMP) and classifies the patient as Normal or Critical. Part of the Luminary (Lumy) graduation project.

## Files
- `main_vi.py` — FastAPI app
- `luminary_vital_signs_best.pth.zip` — trained model checkpoint

## Run locally
```bash
pip install -r requirements.txt
uvicorn main_vi:app --reload --port 8000
```
Then open `http://127.0.0.1:8000/docs` to test.

## Endpoint
- `POST /analyze-vitals` — JSON body with 20 sequential readings, returns Normal / Critical + probabilities.
