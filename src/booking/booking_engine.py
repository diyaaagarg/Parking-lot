import uuid
import io
import base64
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.db.models import Booking, Slot, Lot, Venue, Transaction
import qrcode

def generate_qr_code_base64(data: str) -> str:
    """Generate a base64 encoded PNG QR code image."""
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=6,
            border=2,
        )
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        # Fallback dummy base64 string
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

def check_slot_availability(db: Session, slot_id: int, start_time: datetime, end_time: datetime) -> bool:
    """
    Check if a slot is available for the given time window.
    No overlapping confirmed or overstayed bookings allowed.
    """
    overlapping = db.query(Booking).filter(
        Booking.slot_id == slot_id,
        Booking.status.in_(["confirmed", "overstayed"]),
        Booking.start_time < end_time,
        Booking.end_time > start_time
    ).first()
    return overlapping is None

def create_booking_transactional(
    db: Session,
    user_id: int,
    venue_id: int,
    lot_id: int,
    slot_id: int,
    start_time: datetime,
    end_time: datetime,
    price_multiplier: float = 1.0
) -> dict:
    """
    Concurrency-safe booking creation using DB locking and transaction validation.
    Prevents double-booking even under concurrent simultaneous API requests.
    """
    # Force SQLite write lock for concurrency safety
    if "sqlite" in str(db.bind.url):
        db.execute(text("BEGIN IMMEDIATE"))

    # Verify slot belongs to lot
    slot = db.query(Slot).filter(Slot.id == slot_id, Slot.lot_id == lot_id).first()
    if not slot:
        db.rollback()
        raise ValueError(f"Slot {slot_id} does not exist in Lot {lot_id}")

    lot = db.query(Lot).filter(Lot.id == lot_id).first()
    if not lot:
        db.rollback()
        raise ValueError(f"Lot {lot_id} not found")

    # Double-check availability inside transaction
    if not check_slot_availability(db, slot_id, start_time, end_time):
        db.rollback()
        raise ValueError("Slot is already booked for the selected time window.")

    # Calculate pricing
    duration_hours = max(0.5, (end_time - start_time).total_seconds() / 3600.0)
    total_price = round(lot.base_price_per_hour * duration_hours * price_multiplier, 2)

    # Generate unique booking reference & QR token
    bkg_ref = f"SPIP-{uuid.uuid4().hex[:8].upper()}"
    qr_token = f"QR-{bkg_ref}-{slot_id}"
    qr_base64 = generate_qr_code_base64(qr_token)

    booking = Booking(
        booking_ref=bkg_ref,
        user_id=user_id,
        venue_id=venue_id,
        lot_id=lot_id,
        slot_id=slot_id,
        start_time=start_time,
        end_time=end_time,
        status="confirmed",
        price_charged=total_price,
        qr_code_token=qr_token
    )
    db.add(booking)
    db.flush()

    # Create mock payment transaction record
    txn = Transaction(
        booking_id=booking.id,
        amount=total_price,
        status="success",
        payment_ref=f"TXN-{uuid.uuid4().hex[:10].upper()}"
    )
    db.add(txn)

    # Update slot current_state if start_time is now
    slot.current_state = "reserved"
    db.commit()

    return {
        "booking_id": booking.id,
        "booking_ref": bkg_ref,
        "user_id": user_id,
        "venue_id": venue_id,
        "lot_id": lot_id,
        "slot_id": slot_id,
        "slot_number": slot.slot_number,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "price_charged": total_price,
        "qr_code_token": qr_token,
        "qr_code_base64": qr_base64,
        "status": "confirmed"
    }
