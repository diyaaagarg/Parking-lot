from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.db.models import Slot, Booking, OccupancyEvent

SLOT_STATES = ["vacant", "reserved", "occupied", "overstayed"]

def reconcile_slot_state(db: Session, slot_id: int, is_cv_occupied: bool) -> str:
    """
    Reconciles physical CV occupancy status with DB logical booking status.
    Transitions: vacant -> reserved -> occupied -> overstayed
    """
    slot = db.query(Slot).filter(Slot.id == slot_id).first()
    if not slot:
        return "vacant"

    now = datetime.now(timezone.utc)

    # Check active bookings for this slot
    active_booking = db.query(Booking).filter(
        Booking.slot_id == slot_id,
        Booking.status.in_(["confirmed", "overstayed"]),
        Booking.start_time <= now,
        Booking.end_time >= now
    ).first()

    expired_booking_still_here = db.query(Booking).filter(
        Booking.slot_id == slot_id,
        Booking.status.in_(["confirmed", "overstayed"]),
        Booking.end_time < now
    ).order_by(Booking.end_time.desc()).first()

    new_state = slot.current_state

    if is_cv_occupied:
        if expired_booking_still_here and is_cv_occupied:
            new_state = "overstayed"
            if expired_booking_still_here.status != "overstayed":
                expired_booking_still_here.status = "overstayed"
        elif active_booking:
            new_state = "occupied"
        else:
            # Walk-in driver occupied slot
            new_state = "occupied"
    else:
        # Physical CV sees no car
        upcoming_reservation = db.query(Booking).filter(
            Booking.slot_id == slot_id,
            Booking.status == "confirmed",
            Booking.start_time <= now,
            Booking.end_time >= now
        ).first()

        if upcoming_reservation:
            new_state = "reserved"
        else:
            new_state = "vacant"

    if slot.current_state != new_state:
        slot.current_state = new_state
        db.commit()

    return new_state
