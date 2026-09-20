import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sklearn.ensemble import HistGradientBoostingRegressor
from src.forecasting.features import extract_lot_features

class MLLotForecaster:
    def __init__(self, lot_id: int):
        self.lot_id = lot_id

    def train_and_predict(self, db: Session, horizon_minutes: int = 60) -> dict:
        df = extract_lot_features(db, self.lot_id)
        if df.empty or len(df) < 10:
            return {
                "lot_id": self.lot_id,
                "model_name": "GradientBoosting",
                "predicted_occupancy_pct": 50.0,
                "status": "insufficient_data"
            }

        # Create lag features
        df['lag_15'] = df['occupancy_percentage'].shift(1)
        df['lag_30'] = df['occupancy_percentage'].shift(2)
        df['lag_60'] = df['occupancy_percentage'].shift(4)
        df_clean = df.dropna().copy()

        if len(df_clean) < 5:
            df_clean = df.fillna(0)

        features = ['hour', 'minute', 'day_of_week', 'is_weekend', 'weather_rain', 'event_spike', 'lag_15', 'lag_30', 'lag_60']
        X = df_clean[features]
        y = df_clean['occupancy_percentage']

        model = HistGradientBoostingRegressor(random_state=42, max_iter=100)
        model.fit(X, y)

        # Get latest features for prediction
        latest_occ = float(df['occupancy_percentage'].iloc[-1])
        lag30_val = float(df['occupancy_percentage'].iloc[-2]) if len(df) > 1 else latest_occ
        lag60_val = float(df['occupancy_percentage'].iloc[-4]) if len(df) > 3 else latest_occ

        latest_row = X.iloc[-1:].copy()
        latest_row['lag_15'] = latest_occ
        latest_row['lag_30'] = lag30_val
        latest_row['lag_60'] = lag60_val

        pred_pct = float(model.predict(latest_row)[0])
        pred_pct = max(0.0, min(100.0, pred_pct))
        total_slots = float(df['total_slots'].iloc[-1])
        pred_slots = int(round((pred_pct / 100.0) * total_slots))

        return {
            "lot_id": self.lot_id,
            "model_name": "GradientBoosting (ML)",
            "predicted_occupancy_pct": round(pred_pct, 1),
            "predicted_occupied_slots": pred_slots,
            "total_slots": int(total_slots)
        }
