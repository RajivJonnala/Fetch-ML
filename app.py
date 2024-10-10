from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn as nn  # Ensure nn is imported
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('data_daily.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Aggregate monthly data
monthly_data = data.resample('ME').sum()

# Prepare the data
monthly_data['Month'] = np.arange(1, len(monthly_data) + 1)
X = monthly_data['Month'].values.reshape(-1, 1)
y = monthly_data['Receipt_Count'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load the trained model
class LogarithmicRegression(nn.Module):
    def __init__(self):
        super(LogarithmicRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        x = torch.clamp(x, min=0)
        return self.linear(torch.log(x + 1))

model = LogarithmicRegression()
model.load_state_dict(torch.load('logarithmic_model.pth'))
model.eval()

# Initialize Flask app
app = Flask(__name__)

def predict_receipts(months):
    months_scaled = scaler.transform(months.reshape(-1, 1))
    with torch.no_grad():
        predicted_receipts = model(torch.tensor(months_scaled, dtype=torch.float32)).numpy()
    return predicted_receipts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the number of months to predict from the user
    num_months = int(request.form['num_months'])
    months = np.arange(len(monthly_data) + 1, len(monthly_data) + num_months + 1)
    
    predicted_receipts = predict_receipts(months)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_data.index, monthly_data['Receipt_Count'], marker='o', label='Historical Data')
    
    # Extend the x-axis for predictions
    extended_dates = pd.date_range(start=monthly_data.index[-1] + pd.DateOffset(months=1), 
                                   periods=num_months, freq='ME')
    plt.plot(extended_dates, predicted_receipts, marker='x', color='r', label='Predicted Receipts')
    
    plt.xlabel('Date')
    plt.ylabel('Number of Receipts')
    plt.title('Predicted Receipts')
    plt.legend()
    plt.grid()

    # Save the plot
    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    return jsonify({'plot_url': plot_path})

if __name__ == '__main__':
    app.run(debug=True)
