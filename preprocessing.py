import json
import pandas as pd
import numpy as np

# Load your time-series JSON
try:
    with open('acndata_full_timeseries_1.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: acndata_full_timeseries_1.json not found.")
    exit()

all_points = []

for session in data:
    # STEP 1: Removed sessions where user input values were missing
    user_inputs_list = session.get('userInputs')
    if user_inputs_list is None or len(user_inputs_list) == 0:
        continue
        
    user_input = user_inputs_list[0]
    ed = user_input.get('kWhRequested')
    ma = user_input.get('minutesAvailable')
    total_kwh = session.get('kWhDelivered')
    req_dept = user_input.get('requestedDeparture')

    # STEP 2: Removed rows (within sessions) that had null values
    if any(v is None for v in [ed, ma, total_kwh, req_dept]):
        continue

    # Extract nested time-series objects
    curr_obj = session.get('chargingCurrent')
    pilot_obj = session.get('pilotSignal')
    if not curr_obj or not pilot_obj: continue

    current_series = curr_obj.get('current', [])
    pilot_series = pilot_obj.get('pilot', [])
    timestamps_series = curr_obj.get('timestamps', [])

    if not current_series or len(current_series) != len(timestamps_series):
        continue

    # STEP 4: Derive kWhDeliveredPerTimeStamp (y)
    # Distribute total energy across all non-zero steps
    active_steps = sum(1 for c in current_series if c > 0)
    if active_steps == 0: continue
    kwh_per_step = total_kwh / active_steps

    for i in range(len(current_series)):
        # STEP 3: Remove ONLY if current is exactly 0.0
        # This retains the "trickle charge" points that increase your count
        if current_series[i] == 0:
            continue

        all_points.append({
            'sessionID': session.get('sessionID'),
            'stationID': session.get('stationID'),
            'siteID': session.get('siteID'),
            'kWhRequested': ed,
            'minutesAvailable': ma,
            'requestedDeparture': req_dept,
            'chargingCurrent': current_series[i],
            'pilotSignal': pilot_series[i] if i < len(pilot_series) else 0,
            'timestamp': timestamps_series[i],
            'kWhDeliveredPerTimeStamp': kwh_per_step
        })

# Create DataFrame
df = pd.DataFrame(all_points)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- RESAMPLING TO 1-MINUTE ---
resampled_list = []
for sess_id, group in df.groupby('sessionID'):
    # Resample to 1-minute bins and use 'ffill' to ensure continuity
    resampled_group = group.resample('30s', on='timestamp').agg({
        'stationID': 'first',
        'siteID': 'first',
        'kWhRequested': 'first',
        'minutesAvailable': 'first',
        'requestedDeparture': 'first',
        'chargingCurrent': 'mean',
        'pilotSignal': 'mean',
        'kWhDeliveredPerTimeStamp': 'sum'
    }).ffill().reset_index()
    
    resampled_group['sessionID'] = sess_id
    resampled_list.append(resampled_group)

final_df = pd.concat(resampled_list, ignore_index=True)

# Final step: Only drop rows that are still NaN after ffill
final_df = final_df.dropna()

# Save the finalized dataset
final_df.to_csv('preprocessed_ev_data.csv', index=False)

print(f"--- PREPROCESSING COMPLETE ---")
print(f"Total points extracted: {len(final_df)}")
print(f"Unique sessions processed: {final_df['sessionID'].nunique()}")