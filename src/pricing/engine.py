from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.db.models import Lot
from src.forecasting.prophet_model import ProphetLotForecaster

def calculate_dynamic_price(db: Session, lot_id: int, target_time: datetime = None, target_datetime: datetime = None) -> dict:
    """
    Elasticity-based dynamic pricing algorithm.
    Calculates recommended hourly price multiplier and rate based on predicted demand.
    """
    if target_time is None and target_datetime is not None:
        target_time = target_datetime

    lot = db.query(Lot).filter(Lot.id == lot_id).first()
    if not lot:
        return {"error": f"Lot {lot_id} not found"}

    if target_time is None:
        target_time = datetime.now(timezone.utc)

    forecaster = ProphetLotForecaster(lot_id)
    prediction = forecaster.predict_future(db, horizon_minutes=30)
    pred_occ_pct = prediction.get("predicted_occupancy_pct", 50.0)

    # Elasticity demand curve multiplier calculation
    occ_ratio = pred_occ_pct / 100.0

    if occ_ratio < 0.50:
        multiplier = 1.0
        surge_level = "Normal"
    elif occ_ratio < 0.75:
        multiplier = 1.0 + 0.5 * ((occ_ratio - 0.50) / 0.25)
        surge_level = "Moderate"
    elif occ_ratio < 0.90:
        multiplier = 1.5 + 0.5 * ((occ_ratio - 0.75) / 0.15)
        surge_level = "High"
    else:
        multiplier = 2.0 + 0.5 * min(1.0, (occ_ratio - 0.90) / 0.10)
        surge_level = "Peak Surge"

    # Peak hour surge boost (5 PM - 9 PM)
    if 17 <= target_time.hour <= 21:
        multiplier = min(2.5, multiplier * 1.15)

    base_price = float(lot.base_price_per_hour)
    recommended_price = round(base_price * multiplier, 2)
    estimated_revenue_uplift_pct = round((multiplier - 1.0) * 85.0, 1)

    return {
        "lot_id": lot.id,
        "lot_name": lot.name,
        "base_price_per_hour": base_price,
        "predicted_occupancy_pct": pred_occ_pct,
        "surge_level": surge_level,
        "price_multiplier": round(multiplier, 2),
        "recommended_price_per_hour": recommended_price,
        "estimated_revenue_uplift_pct": estimated_revenue_uplift_pct
    }
