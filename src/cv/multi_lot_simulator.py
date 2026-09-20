import json
import time
import random
from datetime import datetime, timezone
import numpy as np
import cv2
from sqlalchemy.orm import Session
from src.db.database import SessionLocal
from src.db.models import Lot, Slot, OccupancyEvent
from src.booking.slot_state_machine import reconcile_slot_state

def simulate_multi_lot_step():
    """
    Simulates a CV frame detection step across all active parking lots,
    updating slot states and logging time-series occupancy events.
    """
    db = SessionLocal()
    try:
        lots = db.query(Lot).all()
        now_iso = datetime.now(timezone.utc)
        results_summary = []

        for lot in lots:
            slots = db.query(Slot).filter(Slot.lot_id == lot.id).all()
            total_slots = len(slots)
            occupied_count = 0

            for slot in slots:
                # Simulate physical presence based on current state or random pattern
                # If reserved/occupied, higher probability of vehicle present
                if slot.current_state in ["reserved", "occupied"]:
                    is_cv_occupied = random.random() < 0.85
                else:
                    is_cv_occupied = random.random() < 0.30

                # Log physical occupancy event
                event = OccupancyEvent(
                    timestamp=now_iso,
                    lot_id=lot.id,
                    slot_id=slot.id,
                    is_occupied=1 if is_cv_occupied else 0
                )
                db.add(event)

                # Reconcile slot state with booking database
                new_state = reconcile_slot_state(db, slot.id, is_cv_occupied)
                if new_state in ["occupied", "overstayed"]:
                    occupied_count += 1

            db.commit()
            occupancy_pct = round((occupied_count / total_slots) * 100, 1) if total_slots > 0 else 0
            results_summary.append({
                "lot_id": lot.id,
                "lot_name": lot.name,
                "occupied_slots": occupied_count,
                "total_slots": total_slots,
                "occupancy_percentage": occupancy_pct
            })

        return results_summary
    except Exception as e:
        db.rollback()
        print(f"[Multi-Lot Simulator Error] {e}")
        return []
    finally:
        db.close()

if __name__ == "__main__":
    print("Running Multi-Lot Simulation step...")
    res = simulate_multi_lot_step()
    print(json.dumps(res, indent=2))
