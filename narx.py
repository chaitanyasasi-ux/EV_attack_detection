import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np

class NARXModel(nn.Module):
    def __init__(self, input_dim):
        super(NARXModel, self).__init__()
        # Architecture strictly from the paper: 1 hidden layer, 10 neurons, Sigmoid
        self.network = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# Load the 70/15/15 data
data = joblib.load('processed_data.pkl')
X_train = torch.FloatTensor(np.array(data['X_train'], dtype=np.float32))
Y_train = torch.FloatTensor(np.array(data['Y_train'], dtype=np.float32)).view(-1, 1)
X_val = torch.FloatTensor(np.array(data['X_val'], dtype=np.float32))
Y_val = torch.FloatTensor(np.array(data['Y_val'], dtype=np.float32)).view(-1, 1)

model = NARXModel(X_train.shape[1])
# Using Adam as it is more stable for training than basic SGD in this context
optimizer = optim.Adam(model.parameters(), lr=0.01) 
criterion = nn.MSELoss()

best_val_loss = float('inf')

print("Training NARX (Paper-Accurate Architecture)...")
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
    
    # Save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'narx_model.pth')
        
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Train Loss {loss.item():.6f}, Val Loss {val_loss.item():.6f}")

print(f"NARX Training Complete. Best Val Loss: {best_val_loss:.6f}")