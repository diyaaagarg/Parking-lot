import sqlite3
import pandas as pd
import os
import matplotlib.pyplot as plt

DB_PATH = os.path.join("data", "spip.db")

def load_and_prep_data():
    conn = sqlite3.connect(DB_PATH)
    
    # Extract data and convert timestamp string to datetime objects
    df = pd.read_sql_query("SELECT timestamp, slot_id, is_occupied FROM occupancy_events", conn)
    conn.close()
    
    if df.empty:
        print("Database is empty. Run inference to generate data.")
        return
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Feature Engineering for ML
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Calculate overall parking lot occupancy rate per minute
    # Grouping by time blocks allows us to forecast the entire lot's status
    df_grouped = df.groupby([pd.Grouper(key='timestamp', freq='1min')]).agg(
        total_occupied=('is_occupied', 'sum'),
        total_slots=('slot_id', 'nunique')
    ).reset_index()
    
    df_grouped['occupancy_percentage'] = (df_grouped['total_occupied'] / df_grouped['total_slots']) * 100
    
    print("\n--- ML Training Data Prepared ---")
    print(df_grouped.head())
    
    return df_grouped

if __name__ == "__main__":
    prep_data = load_and_prep_data()