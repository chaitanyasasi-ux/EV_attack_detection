import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ev_training_data.csv')

# Pick one session to look at (let's just take the first unique ID)
first_session_id = df['session_id'].unique()[0]
sample_session = df[df['session_id'] == first_session_id]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(sample_session['Time_Step'], sample_session['kWh_Delivered_Actual'], 
         label='Reconstructed Energy Flow', color='blue', linewidth=2)

plt.title(f"Charging Profile for Session: {first_session_id}")
plt.xlabel("Time Step (Minutes)")
plt.ylabel("Energy Delivered (kWh)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Save the image for your presentation
plt.savefig('charging_profile_sample.png')
plt.show()

print("Plot saved as charging_profile_sample.png")