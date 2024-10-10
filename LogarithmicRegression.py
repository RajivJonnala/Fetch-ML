import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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

# Create training datasets
X_train = X_scaled[:-1]
y_train = y[:-1]

# Prepare input for prediction for 2022
months_2022 = np.arange(len(monthly_data) + 1, len(monthly_data) + 13).reshape(-1, 1)
X_2022_scaled = scaler.transform(months_2022)

# Define the Logarithmic Regression model
class LogarithmicRegression(nn.Module):
    def __init__(self):
        super(LogarithmicRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        # Ensure input is valid for log
        x = torch.clamp(x, min=0)  # Clamp to avoid negative values
        return self.linear(torch.log(x + 1))  # Adding 1 to avoid log(0)

# Initialize model, loss function, and optimizer
model = LogarithmicRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    # Convert to tensors
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #if (epoch + 1) % 100 == 0:
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prediction for 2022
model.eval()
with torch.no_grad():
    predicted_receipts = model(torch.tensor(X_2022_scaled, dtype=torch.float32)).numpy()



# Optional: Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data['Receipt_Count'], marker='o', label='Historical Data')
plt.xticks(rotation=45)
for i in range(12):
    if not np.isnan(predicted_receipts[i]):
        plt.text(monthly_data.index[-1] + pd.DateOffset(months=i + 1), 
         predicted_receipts[i], 
         str(int(predicted_receipts[i].item())),  
         ha='center')

# Extend the x-axis for 2022
extended_dates = pd.date_range(start=monthly_data.index[-1] + pd.DateOffset(months=1), 
                               periods=12, freq='ME')
plt.plot(extended_dates, predicted_receipts, marker='x', color='r', label='Predicted 2022 Receipts')
plt.xlabel('Date')
plt.ylabel('Number of Receipts')
plt.title('Scanned Receipts: Historical Data and Predictions for 2022 (Logarithmic Regression)')
plt.legend()
plt.grid()
plt.show()

# Save the trained model
torch.save(model.state_dict(), 'logarithmic_model.pth')