import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('adversarial_ev_dataset.csv')

# Encoding and Scaling
le_station, le_site = LabelEncoder(), LabelEncoder()
df['stationID'] = le_station.fit_transform(df['stationID'])
df['siteID'] = le_site.fit_transform(df['siteID'])

features = ['stationID', 'siteID', 'kWhRequested', 'MA_Sensed', 'pilotSignal']
target = 'kWhDeliveredPerTimeStamp'

scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
df[features] = scaler_x.fit_transform(df[features])
df[[target]] = scaler_y.fit_transform(df[[target]])

joblib.dump(scaler_x, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# Adjusted memory_order to 4 for 30s resampling if you chose to update it
def create_stratified_sequences(data, feature_cols, target_col, memory_order=4):
    X, Y, labels, sensed_y, session_ids = [], [], [], [], []
    
    for sess_id, group in data.groupby('sessionID'):
        group = group.reset_index(drop=True)
        if len(group) <= memory_order: continue
            
        for i in range(memory_order, len(group)):
            exo = group.loc[i, feature_cols].values
            past_y = group.loc[i-memory_order:i-1, target_col].values
            
            X.append(np.concatenate([exo, past_y]))
            Y.append(group.loc[i, target_col])
            labels.append(group.loc[i, 'Attack_Label'])
            sensed_y.append(group.loc[i, target_col])
            session_ids.append(sess_id) 

    return np.array(X), np.array(Y), np.array(labels), np.array(sensed_y), np.array(session_ids)

X, Y, labels, sensed_y, s_ids = create_stratified_sequences(df, features, target, memory_order=4)

# --- SESSION-BASED STRATIFIED SPLIT (70/15/15) ---

# 1. Get unique session IDs and their corresponding attack labels
unique_ids = np.unique(s_ids)
id_labels = [df[df['sessionID'] == sid]['Attack_Label'].max() for sid in unique_ids]

# 2. First Split: Separate 70% for Training
train_ids, temp_ids, train_labels, temp_labels = train_test_split(
    unique_ids, id_labels, test_size=0.30, stratify=id_labels, random_state=42
)

# 3. Second Split: Split the remaining 30% into two equal halves (15% Val, 15% Test)
val_ids, test_ids = train_test_split(
    temp_ids, test_size=0.50, stratify=temp_labels, random_state=42
)

# 4. Create Masks for the sequences
train_mask = np.isin(s_ids, train_ids)
val_mask = np.isin(s_ids, val_ids)
test_mask = np.isin(s_ids, test_ids)

data_bundle = {
    'X_train': X[train_mask], 'Y_train': Y[train_mask],
    'X_val': X[val_mask], 'Y_val': Y[val_mask],
    'X_test': X[test_mask], 'Y_test': Y[test_mask],
    'labels_test': labels[test_mask],
    'Y_sensed_test': sensed_y[test_mask]
}

joblib.dump(data_bundle, 'processed_data.pkl')

print(f"--- 70/15/15 Split Complete ---")
print(f"Train Samples: {len(data_bundle['X_train'])}")
print(f"Val Samples:   {len(data_bundle['X_val'])}")
print(f"Test Samples:  {len(data_bundle['X_test'])}")