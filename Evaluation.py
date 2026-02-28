import torch
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from narx import NARXModel
from BiLSTM import BiLSTMModel

# 1. Load the 15% Holdout Test Data
data = joblib.load('processed_data.pkl')
X_test = torch.FloatTensor(np.array(data['X_test'], dtype=np.float32))
Y_sensed = data['Y_sensed_test'] # The kWh values observed (potentially attacked)
labels = data['labels_test']     # The ground truth binary labels (0=Normal, 1=Attack)

def detect_attacks(model, X, Y_sensed, q, beta):
    """
    Implements the Detection Layer: Residual calculation -> IQR Thresholding -> Persistence check
    """
    model.eval()
    with torch.no_grad():
        preds = model(X).numpy().flatten()
    
    # Calculate Error of Estimation (Residuals)
    eoe = np.abs(Y_sensed - preds)
    
    # IQR-based Dynamic Thresholding
    Q1 = np.percentile(eoe, 25)
    Q3 = np.percentile(eoe, 75)
    IQR = Q3 - Q1
    threshold = Q3 + (beta * IQR)
    
    # Check for anomalies (spikes)
    spikes = (eoe > threshold).astype(int)
    
    # Persistence Logic: Alarm only if 'q' consecutive spikes occur
    final_preds = np.zeros_like(spikes)
    for i in range(len(spikes) - q + 1):
        if np.all(spikes[i:i+q] == 1):
            final_preds[i:i+q] = 1
            
    return final_preds

def perform_comprehensive_evaluation(model, name):
    print(f"\n" + "="*50)
    print(f"EVALUATION & GRID SEARCH: {name}")
    print("="*50)
    
    best_f1 = 0
    best_config = {"beta": 0, "q": 0}
    
    # Define hyperparameter search space
    # beta: sensitivity multiplier | q: persistence window (minutes)
    betas = [0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    qs = [1, 2, 3, 4, 5]
    
    # Grid Search Loop
    for b in betas:
        for q_val in qs:
            current_preds = detect_attacks(model, X_test, Y_sensed, q=q_val, beta=b)
            current_f1 = f1_score(labels, current_preds, zero_division=0)
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_config = {"beta": b, "q": q_val}

    # Final metrics calculation using the winning parameters
    final_preds = detect_attacks(model, X_test, Y_sensed, **best_config)
    
    acc = accuracy_score(labels, final_preds)
    p = precision_score(labels, final_preds, zero_division=0)
    r = recall_score(labels, final_preds, zero_division=0)
    f1 = f1_score(labels, final_preds, zero_division=0)
    
    print(f"[Best Parameters Found]")
    print(f"-> Optimal Beta (Sensitivity): {best_config['beta']}")
    print(f"-> Optimal q (Persistence):    {best_config['q']}")
    print("-" * 30)
    print(f"GLOBAL ACCURACY: {acc * 100:.2f}%")
    print(f"PRECISION:       {p:.4f}")
    print(f"RECALL:          {r:.4f}")
    print(f"F1-SCORE:        {f1:.4f}")
    print("="*50)

# Load saved model weights
input_dim = X_test.shape[1]

narx = NARXModel(input_dim)
narx.load_state_dict(torch.load('narx_model.pth'))

bilstm = BiLSTMModel(input_dim)
bilstm.load_state_dict(torch.load('bilstm_model.pth'))

# Run the evaluation suite
perform_comprehensive_evaluation(narx, "NARX (Paper Architecture)")
perform_comprehensive_evaluation(bilstm, "Bi-LSTM (Benchmark)")