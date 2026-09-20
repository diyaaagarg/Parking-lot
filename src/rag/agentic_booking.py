import re
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from src.db.models import Venue, Lot, Slot, User
from src.booking.booking_engine import create_booking_transactional, check_slot_availability

def execute_agentic_booking_intent(db: Session, user_id: int, query: str) -> dict:
    """
    Agentic AI Assistant Tool:
    Parses natural language user booking intent, finds matching venue/lot/slot,
    and executes a concurrency-safe booking on behalf of the user.
    """
    query_lower = query.lower()

    # 1. Match Venue
    target_venue = None
    venues = db.query(Venue).all()

    for v in venues:
        if v.name.lower() in query_lower or v.category.lower() in query_lower:
            target_venue = v
            break

    if not target_venue:
        target_venue = venues[0]  # Default to DB City Mall if unspecified

    lot = db.query(Lot).filter(Lot.venue_id == target_venue.id).first()
    if not lot:
        return {"success": False, "message": f"No active parking lot found for venue {target_venue.name}"}

    # 2. Extract Requested Time (default: starting in 15 mins for 2 hours)
    now = datetime.now(timezone.utc)
    start_time = now + timedelta(minutes=15)
    end_time = start_time + timedelta(hours=2)

    # Check for explicit hour in query (e.g. 7 PM, 19:00)
    match_hour = re.search(r'(\d{1,2})\s*(pm|am|:00)?', query_lower)
    if match_hour:
        try:
            hr = int(match_hour.group(1))
            meridiem = match_hour.group(2)
            if meridiem == 'pm' and hr < 12:
                hr += 12
            start_time = now.replace(hour=hr, minute=0, second=0, microsecond=0)
            if start_time < now:
                start_time += timedelta(days=1)
            end_time = start_time + timedelta(hours=2)
        except Exception:
            pass

    # 3. Find an available slot
    slots = db.query(Slot).filter(Slot.lot_id == lot.id).all()
    available_slot = None

    for s in slots:
        if check_slot_availability(db, s.id, start_time, end_time):
            available_slot = s
            break

    if not available_slot:
        return {
            "success": False,
            "message": f"No vacant slots available at {target_venue.name} for the requested time window ({start_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')})."
        }

    # 4. Execute Booking Tool Call
    try:
        booking_result = create_booking_transactional(
            db=db,
            user_id=user_id,
            venue_id=target_venue.id,
            lot_id=lot.id,
            slot_id=available_slot.id,
            start_time=start_time,
            end_time=end_time
        )
        return {
            "success": True,
            "message": (
                f"🎉 Success! Agent reserved Slot #{available_slot.slot_number} at {target_venue.name} "
                f"for {start_time.strftime('%b %d, %I:%M %p')} - {end_time.strftime('%I:%M %p')}.\n"
                f"Booking Reference: {booking_result['booking_ref']} | Charged: ${booking_result['price_charged']}"
            ),
            "booking_details": booking_result
        }
    except Exception as e:
        return {"success": False, "message": f"Booking execution failed: {str(e)}"}
