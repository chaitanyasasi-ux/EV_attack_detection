import acnportal.acndata as acndata
from datetime import datetime
import pytz
import json

api_token = '5ONDIPPc5Frg6sKV6IztZU7NsjZYQ_HymuBsU2OazPs' 
client = acndata.DataClient(api_token)


start = datetime(2020, 12, 1, tzinfo=pytz.utc)
end = datetime(2021, 1, 31, tzinfo=pytz.utc)

print(f"New version of download")

all_sessions = []

try:
  
    sessions_generator = client.get_sessions_by_time('caltech', start=start, end=end, timeseries=True)

    # 2. Loop through one-by-one
    for i, session in enumerate(sessions_generator):
        all_sessions.append(session)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        session_id = session.get('sessionID', 'Unknown')
        print(f"[{timestamp}] Downloaded session {i+1}: ID {session_id}")

    # 3. Save once all are collected
    if all_sessions:
        with open('acndata_full_timeseries_1.json', 'w') as f:
            json.dump(all_sessions, f, indent=4, default=str)
        print(f"\n--- SUCCESS: {len(all_sessions)} sessions saved ---")
    else:
        print("No sessions found in this range.")

except Exception as e:
    print(f"\nError: {e}")