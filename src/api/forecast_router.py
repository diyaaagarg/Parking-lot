from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.db.database import get_db
from src.forecasting.prophet_model import ProphetLotForecaster
from src.forecasting.ml_forecaster import MLLotForecaster
from src.forecasting.evaluator import evaluate_models_for_lot
from src.pricing.engine import calculate_dynamic_price

router = APIRouter(prefix="/forecast", tags=["Forecasting & Dynamic Pricing"])

@router.get("/{lot_id}")
def get_lot_forecast(lot_id: int, horizon_minutes: int = 60, db: Session = Depends(get_db)):
    prophet_fc = ProphetLotForecaster(lot_id)
    p_res = prophet_fc.predict_future(db, horizon_minutes=horizon_minutes)

    ml_fc = MLLotForecaster(lot_id)
    m_res = ml_fc.train_and_predict(db, horizon_minutes=horizon_minutes)

    return {
        "lot_id": lot_id,
        "horizon_minutes": horizon_minutes,
        "prophet_forecast": p_res,
        "ml_forecast": m_res
    }

@router.get("/{lot_id}/eval")
def get_lot_evaluation(lot_id: int, db: Session = Depends(get_db)):
    metrics = evaluate_models_for_lot(db, lot_id)
    return metrics

@router.get("/pricing/recommend/{lot_id}")
def get_pricing_recommendation(lot_id: int, db: Session = Depends(get_db)):
    result = calculate_dynamic_price(db, lot_id)
    return result
