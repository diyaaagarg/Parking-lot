from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.db.database import get_db
from src.db.models import User, Booking, Slot
from src.api.auth import get_current_user
from src.booking.booking_engine import create_booking_transactional
from src.pricing.engine import calculate_dynamic_price

router = APIRouter(prefix="/bookings", tags=["Bookings & Concurrency"])

class BookingReserveRequest(BaseModel):
    venue_id: int
    lot_id: int
    slot_id: int
    start_time_iso: str
    end_time_iso: str

@router.post("/reserve")
def reserve_slot(
    req: BookingReserveRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        start_dt = datetime.fromisoformat(req.start_time_iso.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(req.end_time_iso.replace('Z', '+00:00'))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ISO timestamp format.")

    if start_dt >= end_dt:
        raise HTTPException(status_code=400, detail="Start time must be before end time.")

    # Calculate dynamic pricing multiplier
    pricing = calculate_dynamic_price(db, req.lot_id, start_dt)
    price_mult = pricing.get("price_multiplier", 1.0)

    try:
        result = create_booking_transactional(
            db=db,
            user_id=current_user.id,
            venue_id=req.venue_id,
            lot_id=req.lot_id,
            slot_id=req.slot_id,
            start_time=start_dt,
            end_time=end_dt,
            price_multiplier=price_mult
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Booking error: {str(e)}")

@router.get("/my-bookings")
def list_my_bookings(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    bookings = db.query(Booking).filter(Booking.user_id == current_user.id).order_by(Booking.id.desc()).all()
    results = []
    for b in bookings:
        results.append({
            "id": b.id,
            "booking_ref": b.booking_ref,
            "venue_name": b.venue.name if b.venue else "N/A",
            "lot_name": b.lot.name if b.lot else "N/A",
            "slot_number": b.slot.slot_number if b.slot else N/A,
            "start_time": b.start_time.isoformat(),
            "end_time": b.end_time.isoformat(),
            "status": b.status,
            "price_charged": b.price_charged,
            "qr_code_token": b.qr_code_token,
            "created_at": b.created_at.isoformat()
        })
    return results

@router.post("/{booking_id}/cancel")
def cancel_booking(booking_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    booking = db.query(Booking).filter(Booking.id == booking_id, Booking.user_id == current_user.id).first()
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found.")

    if booking.status == "cancelled":
        raise HTTPException(status_code=400, detail="Booking is already cancelled.")

    booking.status = "cancelled"
    slot = db.query(Slot).filter(Slot.id == booking.slot_id).first()
    if slot and slot.current_state == "reserved":
        slot.current_state = "vacant"

    db.commit()
    return {"message": f"Booking {booking.booking_ref} cancelled successfully.", "refund_amount": booking.price_charged}

@router.post("/{booking_id}/check-in")
def check_in_barrier(booking_id: int, db: Session = Depends(get_db)):
    booking = db.query(Booking).filter(Booking.id == booking_id).first()
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found.")

    slot = db.query(Slot).filter(Slot.id == booking.slot_id).first()
    if slot:
        slot.current_state = "occupied"
        db.commit()

    return {
        "message": f"Check-in verified for {booking.booking_ref}. Barrier opened!",
        "slot_number": slot.slot_number if slot else None,
        "current_slot_state": "occupied"
    }
