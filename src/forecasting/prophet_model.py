import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from src.forecasting.features import extract_lot_features

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

class ProphetLotForecaster:
    def __init__(self, lot_id: int):
        self.lot_id = lot_id

    def predict_future(self, db: Session, horizon_minutes: int = 60) -> dict:
        """
        Predicts lot occupancy percentage and count for 30 and 60 minutes ahead.
        """
        df = extract_lot_features(db, self.lot_id)
        if df.empty or len(df) < 10:
            return {
                "lot_id": self.lot_id,
                "horizon_minutes": horizon_minutes,
                "predicted_occupancy_pct": 50.0,
                "predicted_occupied_slots": 10,
                "model_used": "fallback_baseline",
                "trend": "stable"
            }

        total_slots = float(df['total_slots'].iloc[-1])

        if PROPHET_AVAILABLE:
            try:
                prophet_df = df[['timestamp', 'occupancy_percentage']].rename(
                    columns={'timestamp': 'ds', 'occupancy_percentage': 'y'}
                )
                prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

                model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
                model.fit(prophet_df)

                future = model.make_future_dataframe(periods=int(horizon_minutes / 15), freq='15min')
                forecast = model.predict(future)

                pred_pct = float(forecast['yhat'].iloc[-1])
                pred_pct = max(0.0, min(100.0, pred_pct))
                pred_slots = int(round((pred_pct / 100.0) * total_slots))

                return {
                    "lot_id": self.lot_id,
                    "horizon_minutes": horizon_minutes,
                    "predicted_occupancy_pct": round(pred_pct, 1),
                    "predicted_occupied_slots": pred_slots,
                    "total_slots": int(total_slots),
                    "model_used": "Prophet",
                    "forecast_series": forecast[['ds', 'yhat']].tail(8).to_dict(orient="records")
                }
            except Exception as e:
                print(f"[Prophet Error] {e}. Falling back to ML feature regression.")

        # Fallback regression using time features
        from sklearn.ensemble import RandomForestRegressor
        X = df[['hour', 'minute', 'day_of_week', 'is_weekend', 'event_spike']]
        y = df['occupancy_percentage']

        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X, y)

        future_time = datetime.now(timezone.utc) + timedelta(minutes=horizon_minutes)
        future_feats = pd.DataFrame([{
            "hour": future_time.hour,
            "minute": future_time.minute,
            "day_of_week": future_time.weekday(),
            "is_weekend": 1 if future_time.weekday() in [5, 6] else 0,
            "event_spike": 1.25 if future_time.weekday() in [5, 6] and 16 <= future_time.hour <= 21 else 1.0
        }])

        pred_pct = float(rf.predict(future_feats)[0])
        pred_pct = max(0.0, min(100.0, pred_pct))
        pred_slots = int(round((pred_pct / 100.0) * total_slots))

        return {
            "lot_id": self.lot_id,
            "horizon_minutes": horizon_minutes,
            "predicted_occupancy_pct": round(pred_pct, 1),
            "predicted_occupied_slots": pred_slots,
            "total_slots": int(total_slots),
            "model_used": "RandomForest_Fallback"
        }
