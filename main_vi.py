from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import numpy as np

# 1. Initialize FastAPI app
app = FastAPI(title="Luminary Vitals Analysis API")
device = torch.device("cpu")

# ==========================================
# 2. DEFINE THE NEURAL NETWORK ARCHITECTURE
# ==========================================
class VitalSignsMLPLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, dropout=0.30):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        batch_size, seq_len, features = x.size()
        x = x.reshape(-1, features)
        x = self.mlp(x)
        x = x.reshape(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        final_output = lstm_out[:, -1, :]
        return self.classifier(final_output)

# ==========================================
# 3. LOAD THE MODEL AND UNPACK THE DICT
# ==========================================
model = VitalSignsMLPLSTM(input_dim=4, hidden_dim=64, dropout=0.30)

# Make sure this matches the exact name of your file
MODEL_PATH = "luminary_vital_signs_best.pth.zip" 

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # CHANGE THIS LINE BELOW
    best_threshold = 0.70  # <-- Hardcode it to match your notebook
    print("Model loaded successfully.")

except Exception as e:
    print(f"Failed to load model. Error: {e}")
    best_threshold = 0.70 # Fallback

# ==========================================
# 4. DEFINE THE INCOMING JSON STRUCTURE
# ==========================================
# Represents a single reading at a specific second
class VitalReading(BaseModel):
    HR: float
    SPO2: float
    RESP: float
    TEMP: float

# Represents the full payload (an array of 20 readings)
class PatientSequence(BaseModel):
    readings: List[VitalReading]

# ==========================================
# 5. THE PREDICTION ENDPOINT
# ==========================================
@app.post("/analyze-vitals")
def analyze_vitals(payload: PatientSequence):
    # Guardrail: Ensure exactly 20 time steps are provided
    if len(payload.readings) != 20:
        raise HTTPException(status_code=400, detail="Exactly 20 sequential readings are required.")
    
    try:
        # Convert JSON readings to a 2D numpy array [20, 4]
        sequence_data = []
        for r in payload.readings:
            sequence_data.append([r.HR, r.SPO2, r.RESP, r.TEMP])
        
        raw_sequence = np.array(sequence_data, dtype=np.float32)
        
        # REMOVED THE SCALER: We now pass the raw numbers directly to the tensor
        # exactly like the manual testing cells in your Jupyter Notebook!
        tensor_sequence = torch.tensor(raw_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Run Inference
        with torch.no_grad():
            logits = model(tensor_sequence)
            probs = torch.softmax(logits, dim=1)
            
            prob_normal = probs[0][0].item()
            prob_critical = probs[0][1].item()
            
        prediction = "Critical" if prob_critical >= best_threshold else "Normal"
        
        return {
            "status": "success",
            "prediction": prediction,
            "probability_critical": round(prob_critical, 4),
            "probability_normal": round(prob_normal, 4),
            "threshold_used": best_threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Vitals Analysis API is running!"}