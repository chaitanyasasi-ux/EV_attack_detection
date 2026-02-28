import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np

# 1. Architecture Setup
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim):
        super(BiLSTMModel, self).__init__()
        # Paper comparison spec: 10 hidden units, bidirectional
        self.lstm = nn.LSTM(input_dim, 10, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(10 * 2, 1) # 2 directions * 10 units
        
    def forward(self, x):
        # LSTM needs 3D input: [batch, seq_len=1, features]
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        # We take the output of the last (and only) time step
        return self.fc(lstm_out[:, -1, :])

# 2. Data Loading (70/15/15)
data = joblib.load('processed_data.pkl')
X_train = torch.FloatTensor(np.array(data['X_train'], dtype=np.float32))
Y_train = torch.FloatTensor(np.array(data['Y_train'], dtype=np.float32)).view(-1, 1)

X_val = torch.FloatTensor(np.array(data['X_val'], dtype=np.float32))
Y_val = torch.FloatTensor(np.array(data['Y_val'], dtype=np.float32)).view(-1, 1)

# 3. Model Initialization & Weighted Loss
model = BiLSTMModel(X_train.shape[1])

# POS_WEIGHT: To fix the 2% Recall. 
# We tell the model that missing an attack is 5x more "expensive" than a false alarm.
pos_weight = torch.tensor([5.0])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
optimizer = optim.Adam(model.parameters(), lr=0.01)

best_val_loss = float('inf')

# 4. Training Loop
print("Training Bi-LSTM with Weighted Loss for Recall Improvement...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()
    
    # Validation Phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, Y_val)
    
    # Early Saving based on Val Loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'bilstm_model.pth')
        
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Train Loss {loss.item():.6f}, Val Loss {val_loss.item():.6f}")

print(f"Training Complete. Best Validation Loss: {best_val_loss:.6f}")