import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from src.db.models import OccupancyEvent, Lot

def extract_lot_features(db: Session, lot_id: int) -> pd.DataFrame:
    """
    Extracts time-series occupancy records for a lot and engineers exogenous features
    such as time-of-day, day-of-week, weekend flags, weather signals, and event spikes.
    """
    events = db.query(OccupancyEvent).filter(OccupancyEvent.lot_id == lot_id).all()
    if not events:
        return pd.DataFrame()

    data = [{"timestamp": e.timestamp, "is_occupied": e.is_occupied, "slot_id": e.slot_id} for e in events]
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Aggregate by 15-minute intervals
    df_agg = df.groupby(pd.Grouper(key='timestamp', freq='15min')).agg(
        occupied_slots=('is_occupied', 'sum'),
        total_slots=('slot_id', 'nunique')
    ).reset_index()

    # Fill missing timestamps
    df_agg['total_slots'] = df_agg['total_slots'].replace(0, np.nan).ffill().bfill()
    df_agg['total_slots'] = df_agg['total_slots'].fillna(20)
    df_agg['occupied_slots'] = df_agg['occupied_slots'].fillna(0)
    df_agg['occupancy_percentage'] = (df_agg['occupied_slots'] / df_agg['total_slots']) * 100.0

    # Feature Engineering
    df_agg['hour'] = df_agg['timestamp'].dt.hour
    df_agg['minute'] = df_agg['timestamp'].dt.minute
    df_agg['day_of_week'] = df_agg['timestamp'].dt.dayofweek
    df_agg['is_weekend'] = df_agg['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)

    # Simulated exogenous signals (weather & local event calendar)
    df_agg['weather_rain'] = df_agg['hour'].apply(lambda h: 1 if h in [17, 18] and np.random.random() < 0.2 else 0)
    df_agg['event_spike'] = df_agg.apply(
        lambda r: 1.25 if r['is_weekend'] == 1 and 16 <= r['hour'] <= 21 else 1.0,
        axis=1
    )

    return df_agg
