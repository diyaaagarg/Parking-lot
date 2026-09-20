import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.db.database import SessionLocal, Base, engine
from src.db.models import User, Venue, Lot, Slot, Booking, OccupancyEvent, Transaction
from src.api.auth import hash_password, verify_password, create_access_token, get_current_user
from src.booking.booking_engine import create_booking_transactional, check_slot_availability
from src.booking.slot_state_machine import reconcile_slot_state
from src.forecasting.prophet_model import ProphetLotForecaster
from src.forecasting.evaluator import evaluate_models_for_lot
from src.pricing.engine import calculate_dynamic_price
from src.rag.grounded_advisory import answer_grounded_advisory_query
from src.rag.agentic_booking import execute_agentic_booking_intent

class TestSPIPPlatform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n==================================================")
        print("  Running SPIP v2 Platform Automated Verification")
        print("==================================================")
        cls.db = SessionLocal()

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_01_authentication(self):
        print("\n[Test 1] Testing Password Hashing & JWT Auth Token...")
        raw_pw = "secret123"
        hashed = hash_password(raw_pw)
        self.assertTrue(verify_password(raw_pw, hashed))
        self.assertFalse(verify_password("wrong_pw", hashed))

        token = create_access_token({"sub": "driver@spip.com", "role": "driver"})
        self.assertIsNotNone(token)
        print("  -> Auth test passed!")

    def test_02_slot_availability_and_booking_concurrency(self):
        print("\n[Test 2] Testing Concurrency-Safe Booking Engine & Lock...")
        user = self.db.query(User).filter_by(email="driver@spip.com").first()
        slot = self.db.query(Slot).first()
        self.assertIsNotNone(user)
        self.assertIsNotNone(slot)

        now = datetime.now(timezone.utc)
        s_time = now + timedelta(days=2, hours=1)
        e_time = s_time + timedelta(hours=2)

        # Clear prior test bookings for this slot to ensure test idempotency
        b_ids = [b.id for b in self.db.query(Booking).filter(Booking.slot_id == slot.id).all()]
        if b_ids:
            self.db.query(Transaction).filter(Transaction.booking_id.in_(b_ids)).delete(synchronize_session=False)
            self.db.query(Booking).filter(Booking.slot_id == slot.id).delete(synchronize_session=False)
            self.db.commit()

        # 1st Booking should succeed
        b1 = create_booking_transactional(
            db=self.db,
            user_id=user.id,
            venue_id=slot.lot.venue_id,
            lot_id=slot.lot_id,
            slot_id=slot.id,
            start_time=s_time,
            end_time=e_time
        )
        self.assertIsNotNone(b1["booking_ref"])
        self.assertIsNotNone(b1["qr_code_base64"])

        # 2nd Overlapping booking for same slot & window MUST fail!
        with self.assertRaises(ValueError):
            create_booking_transactional(
                db=self.db,
                user_id=user.id,
                venue_id=slot.lot.venue_id,
                lot_id=slot.lot_id,
                slot_id=slot.id,
                start_time=s_time + timedelta(minutes=30),
                end_time=e_time + timedelta(minutes=30)
            )
        print("  -> Concurrency double-booking prevention verified successfully!")

    def test_03_slot_state_machine_reconciliation(self):
        print("\n[Test 3] Testing Physical CV vs Logical State Machine...")
        slot = self.db.query(Slot).first()

        # CV detects vehicle physically present
        state = reconcile_slot_state(self.db, slot.id, is_cv_occupied=True)
        self.assertIn(state, ["occupied", "reserved", "overstayed"])

        # CV detects vehicle vacant
        state_vacant = reconcile_slot_state(self.db, slot.id, is_cv_occupied=False)
        self.assertIn(state_vacant, ["vacant", "reserved"])
        print("  -> State Machine reconciliation verified!")

    def test_04_forecasting_and_evaluator(self):
        print("\n[Test 4] Testing Forecasting Models & Evaluator...")
        lot = self.db.query(Lot).first()
        forecaster = ProphetLotForecaster(lot.id)
        fc = forecaster.predict_future(self.db, horizon_minutes=60)
        self.assertIn("predicted_occupancy_pct", fc)

        eval_res = evaluate_models_for_lot(self.db, lot.id)
        self.assertIn("best_performing_model", eval_res)
        print(f"  -> Forecast evaluation completed. Best Model: {eval_res.get('best_performing_model')}")

    def test_05_dynamic_pricing(self):
        print("\n[Test 5] Testing Dynamic Pricing Engine...")
        lot = self.db.query(Lot).first()
        pricing = calculate_dynamic_price(self.db, lot.id)
        self.assertIn("recommended_price_per_hour", pricing)
        self.assertIn("price_multiplier", pricing)
        print(f"  -> Dynamic price calculated: ${pricing['recommended_price_per_hour']}/hr (Multiplier: {pricing['price_multiplier']})")

    def test_06_grounded_rag_advisory(self):
        print("\n[Test 6] Testing Grounded RAG Advisory Q&A...")
        res = answer_grounded_advisory_query(self.db, "What is the parking rate and occupancy at DB City Mall?")
        self.assertTrue(res["grounded_context_used"])
        self.assertIn("DB City", res["answer"])
        print("  -> Grounded RAG response verified!")

    def test_07_agentic_booking(self):
        print("\n[Test 7] Testing Agentic Natural Language Tool-Calling Booking...")
        user = self.db.query(User).filter_by(email="driver@spip.com").first()
        res = execute_agentic_booking_intent(self.db, user.id, "Book a slot for me at PVR Cinemas Plaza for tomorrow at 5 PM")
        self.assertTrue(res["success"])
        self.assertIn("SPIP-", res["booking_details"]["booking_ref"])
        print(f"  -> Agentic booking executed! Ref: {res['booking_details']['booking_ref']}")

if __name__ == "__main__":
    unittest.main()
