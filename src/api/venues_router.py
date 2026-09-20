import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.db.database import get_db
from src.db.models import Venue, Lot, Slot

router = APIRouter(prefix="/venues", tags=["Venues & Slots"])

@router.get("/")
def list_venues(db: Session = Depends(get_db)):
    venues = db.query(Venue).all()
    results = []
    for v in venues:
        total_slots = 0
        occupied_slots = 0
        for l in v.lots:
            slots = db.query(Slot).filter(Slot.lot_id == l.id).all()
            total_slots += len(slots)
            occupied_slots += sum(1 for s in slots if s.current_state in ["occupied", "overstayed"])

        results.append({
            "id": v.id,
            "name": v.name,
            "address": v.address,
            "latitude": v.latitude,
            "longitude": v.longitude,
            "category": v.category,
            "total_capacity": v.total_capacity,
            "live_total_slots": total_slots,
            "live_occupied_slots": occupied_slots,
            "live_occupancy_pct": round((occupied_slots / total_slots * 100), 1) if total_slots > 0 else 0.0
        })
    return results

@router.get("/{venue_id}/occupancy")
def get_venue_occupancy(venue_id: int, db: Session = Depends(get_db)):
    venue = db.query(Venue).filter(Venue.id == venue_id).first()
    if not venue:
        raise HTTPException(status_code=404, detail=f"Venue {venue_id} not found")

    lots_data = []
    for lot in venue.lots:
        slots = db.query(Slot).filter(Slot.lot_id == lot.id).all()
        s_count = len(slots)
        occ = sum(1 for s in slots if s.current_state in ["occupied", "overstayed"])
        res = sum(1 for s in slots if s.current_state == "reserved")

        lots_data.append({
            "lot_id": lot.id,
            "lot_name": lot.name,
            "base_price_per_hour": lot.base_price_per_hour,
            "total_slots": s_count,
            "occupied_slots": occ,
            "reserved_slots": res,
            "vacant_slots": s_count - occ - res,
            "occupancy_pct": round((occ / s_count * 100), 1) if s_count > 0 else 0.0
        })

    return {
        "venue_id": venue.id,
        "venue_name": venue.name,
        "category": venue.category,
        "lots": lots_data
    }

@router.get("/{venue_id}/slots")
def get_venue_slots(venue_id: int, db: Session = Depends(get_db)):
    venue = db.query(Venue).filter(Venue.id == venue_id).first()
    if not venue:
        raise HTTPException(status_code=404, detail=f"Venue {venue_id} not found")

    result_slots = []
    for lot in venue.lots:
        slots = db.query(Slot).filter(Slot.lot_id == lot.id).all()
        for s in slots:
            result_slots.append({
                "slot_id": s.id,
                "lot_id": lot.id,
                "lot_name": lot.name,
                "slot_number": s.slot_number,
                "polygon_coordinates": json.loads(s.polygon_coordinates) if s.polygon_coordinates else [],
                "current_state": s.current_state
            })
    return result_slots
