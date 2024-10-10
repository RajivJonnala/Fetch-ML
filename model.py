import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Load the model
class LogarithmicRegression(nn.Module):
    def __init__(self):
        super(LogarithmicRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        x = torch.clamp(x, min=0)
        return self.linear(torch.log(x + 1))

def load_model():
    model = LogarithmicRegression()
    model.load_state_dict(torch.load('logarithmic_model.pth'))
    model.eval()
    return model

def predict(model, months):
    # Normalize input months
    scaler = StandardScaler()
    months_scaled = scaler.fit_transform(months.reshape(-1, 1))
    with torch.no_grad():
        predicted = model(torch.tensor(months_scaled, dtype=torch.float32)).numpy()
    return predicted
