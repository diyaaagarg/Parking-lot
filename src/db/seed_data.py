import os
import json
import random
from datetime import datetime, timedelta, timezone
from src.config import DB_PATH
from src.api.auth import hash_password
from src.db.database import engine, SessionLocal, Base
from src.db.models import User, Venue, Lot, Slot, OccupancyEvent, KnowledgeDoc

def seed_database(force_reset: bool = True):
    if force_reset and os.path.exists(DB_PATH):
        print(f"Resetting existing database at {DB_PATH} for new schema...")
        try:
            Base.metadata.drop_all(bind=engine)
        except Exception as e:
            print(f"Drop error: {e}")

    print("Initializing Database tables...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # Check if already seeded
        if db.query(User).first() is not None:
            print("Database already seeded. Skipping initial seed.")
            return

        print("Seeding Users...")
        operator = User(
            name="City Parking Admin",
            email="operator@spip.com",
            hashed_password=hash_password("operator123"),
            role="operator"
        )
        driver = User(
            name="Riya Sharma",
            email="driver@spip.com",
            hashed_password=hash_password("driver123"),
            role="driver"
        )
        db.add_all([operator, driver])
        db.commit()

        print("Seeding Venues and Lots...")
        v1 = Venue(
            name="DB City Mall",
            address="Arera Hills, Zone-I, Bhopal",
            latitude=23.2333,
            longitude=77.4333,
            category="mall",
            total_capacity=20
        )
        v2 = Venue(
            name="PVR Cinemas Plaza",
            address="Theater District, Sector 5",
            latitude=23.2450,
            longitude=77.4420,
            category="theater",
            total_capacity=15
        )
        v3 = Venue(
            name="Sector 5 Financial Hub",
            address="Commercial Complex, Downtown",
            latitude=23.2500,
            longitude=77.4500,
            category="downtown",
            total_capacity=25
        )
        db.add_all([v1, v2, v3])
        db.commit()

        lot1 = Lot(venue_id=v1.id, name="DB City Mall Parking - Lot A", total_slots=20, base_price_per_hour=5.0)
        lot2 = Lot(venue_id=v2.id, name="PVR Plaza Basement Lot", total_slots=15, base_price_per_hour=4.0)
        lot3 = Lot(venue_id=v3.id, name="Downtown Sector 5 Surface Lot", total_slots=25, base_price_per_hour=6.0)
        db.add_all([lot1, lot2, lot3])
        db.commit()

        print("Seeding Slots...")
        lots_config = [
            (lot1, 20),
            (lot2, 15),
            (lot3, 25)
        ]

        for lot_obj, count in lots_config:
            for s_idx in range(count):
                x_base = 50 + (s_idx % 5) * 110
                y_base = 50 + (s_idx // 5) * 90
                coords = [
                    [x_base, y_base],
                    [x_base + 90, y_base],
                    [x_base + 90, y_base + 70],
                    [x_base, y_base + 70]
                ]
                slot = Slot(
                    lot_id=lot_obj.id,
                    slot_number=s_idx + 1,
                    polygon_coordinates=json.dumps(coords),
                    current_state="vacant"
                )
                db.add(slot)
        db.commit()

        print("Seeding 7-Day Historical Occupancy Data for Forecasting...")
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=7)

        lots = [lot1, lot2, lot3]
        lot_slots_map = {l.id: db.query(Slot).filter_by(lot_id=l.id).all() for l in lots}

        events = []
        curr_time = start_date
        while curr_time <= now:
            hour = curr_time.hour
            day_of_week = curr_time.weekday()

            if 11 <= hour <= 14 or 17 <= hour <= 20:
                base_occ_ratio = 0.75 + (0.15 * random.random())
            elif 8 <= hour <= 10 or 15 <= hour <= 16:
                base_occ_ratio = 0.45 + (0.15 * random.random())
            elif 21 <= hour <= 23:
                base_occ_ratio = 0.30 + (0.10 * random.random())
            else:
                base_occ_ratio = 0.05 + (0.05 * random.random())

            if day_of_week in [5, 6]:
                base_occ_ratio = min(0.95, base_occ_ratio * 1.25)

            for l in lots:
                slots = lot_slots_map[l.id]
                occupied_count = int(len(slots) * base_occ_ratio)
                occupied_indices = set(random.sample(range(len(slots)), occupied_count))

                for idx, slot in enumerate(slots):
                    is_occ = 1 if idx in occupied_indices else 0
                    events.append(OccupancyEvent(
                        timestamp=curr_time,
                        lot_id=l.id,
                        slot_id=slot.id,
                        is_occupied=is_occ
                    ))

            if len(events) >= 2000:
                db.bulk_save_objects(events)
                events = []

            curr_time += timedelta(minutes=15)

        if events:
            db.bulk_save_objects(events)
        db.commit()

        print("Seeding Operator Knowledge Base Documents for RAG...")
        docs = [
            KnowledgeDoc(
                title="DB City Mall Parking Policy",
                category="policy",
                content=(
                    "DB City Mall Lot A operates 24/7. Base pricing is $5.00/hour. "
                    "Peak hours run from 5:00 PM to 9:00 PM daily, during which dynamic pricing up to $8.50/hour applies. "
                    "Reservations made in advance guarantee a spot for up to 15 minutes past the start time. "
                    "Overstaying past the booked window incurs a 1.5x hourly penalty."
                )
            ),
            KnowledgeDoc(
                title="PVR Cinemas Plaza Parking Rules",
                category="policy",
                content=(
                    "PVR Cinemas Plaza Basement Lot has 15 premium reserved slots. Base rate is $4.00/hour. "
                    "Cinema ticket holders get a 10% discount on pre-booked parking slots. "
                    "Peak demand occurs Friday through Sunday evenings (6:00 PM - 10:00 PM). "
                    "Overstaying beyond 30 minutes after movie end time results in automatic overstay flagging."
                )
            ),
            KnowledgeDoc(
                title="Downtown Sector 5 Parking & Dynamic Pricing Regulations",
                category="pricing_rule",
                content=(
                    "Sector 5 Financial Hub Lot has 25 surface slots. Base rate is $6.00/hour on weekdays. "
                    "Dynamic pricing triggers automatically when lot occupancy exceeds 75% capacity. "
                    "Max peak rate is capped at $12.00/hour by municipal regulations. "
                    "Free parking is allowed between 11:00 PM and 6:00 AM on weekdays."
                )
            ),
            KnowledgeDoc(
                title="SPIP Reservation & Cancellation Guidelines",
                category="venue_info",
                content=(
                    "Drivers can reserve specific parking slots up to 24 hours in advance via the SPIP app. "
                    "Cancellations made at least 30 minutes before the start time receive a full 100% refund. "
                    "Check-in is verified automatically via QR Code scan at the parking barrier or CV camera detection."
                )
            )
        ]
        db.add_all(docs)
        db.commit()

        print("Database Seed Completed Successfully!")

    except Exception as e:
        db.rollback()
        print(f"Error during seeding: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
