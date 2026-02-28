import pandas as pd
import numpy as np

# Load your preprocessed dataset
df = pd.read_csv('preprocessed_ev_data.csv')

def simulate_paper_attacks(df):
    attacked_df = df.copy()
    attacked_df['MA_Sensed'] = attacked_df['minutesAvailable']
    attacked_df['Attack_Label'] = 0 
    
    unique_sessions = attacked_df['sessionID'].unique()
    
    for sess_id in unique_sessions:
        # 18% attack probability as discussed
        if np.random.rand() < 0.18:
            mask = attacked_df['sessionID'] == sess_id
            indices = attacked_df[mask].index
            session_len = len(indices)
            
            if session_len < 12: continue # Need enough points for a ramp

            # Determine attack start and duration
            start_idx = np.random.randint(2, session_len // 3)
            duration = np.random.randint(8, session_len - start_idx) 
            attack_indices = indices[start_idx : start_idx + duration]
            
            # --- NEW: Mixed Attack Profiles ---
            # Step = Paper's fixed phi | Ramp = Stealthy gradual drift
            attack_profile = np.random.choice(['step', 'ramp'])
            
            if attack_profile == 'step':
                phi = np.random.uniform(30, 80)
                attacked_df.loc[attack_indices, 'MA_Sensed'] += phi
            else:
                # Ramp: phi grows linearly from 0 to max_phi
                max_phi = np.random.uniform(60, 120)
                ramp_values = np.linspace(0, max_phi, len(attack_indices))
                attacked_df.loc[attack_indices, 'MA_Sensed'] += ramp_values
            
            attacked_df.loc[attack_indices, 'Attack_Label'] = 1
            # Physical Impact: Halting energy delivery during attack
            attacked_df.loc[attack_indices, 'kWhDeliveredPerTimeStamp'] = 0.0

    return attacked_df

adversarial_df = simulate_paper_attacks(df)
adversarial_df.to_csv('adversarial_ev_dataset.csv', index=False)
print(f"Simulation Complete. Attacked Sessions: {adversarial_df[adversarial_df['Attack_Label']==1]['sessionID'].nunique()}")