import sys
import os
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import base64
from datetime import datetime, timedelta, timezone, time
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy.orm import Session
from src.config import API_BASE_URL
from src.db.database import SessionLocal
from src.db.models import Venue, Lot, Slot, Booking, User, OccupancyEvent, Transaction
from src.api.auth import hash_password, verify_password, create_access_token
from src.booking.booking_engine import create_booking_transactional, check_slot_availability
from src.booking.slot_state_machine import reconcile_slot_state
from src.forecasting.prophet_model import ProphetLotForecaster
from src.forecasting.evaluator import evaluate_models_for_lot
from src.pricing.engine import calculate_dynamic_price
from src.rag.grounded_advisory import answer_grounded_advisory_query
from src.rag.agentic_booking import execute_agentic_booking_intent
from src.rag.report_generator import generate_operator_summary_report
from src.cv.multi_lot_simulator import simulate_multi_lot_step

st.set_page_config(
    page_title="SPIP v2 - Smart Parking Intelligence Platform",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem; }
    .auth-container { max-width: 500px; margin: 0 auto; padding: 2rem; border: 1px solid #E2E8F0; border-radius: 10px; background-color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

# Session State Initialization for Auth
if "authenticated_user" not in st.session_state:
    st.session_state["authenticated_user"] = None

# Header Banner
st.markdown("<div class='main-header'>🅿️ Smart Parking Intelligence Platform (SPIP v2)</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Multi-Lot Occupancy Detection, Predictive Forecasting, Dynamic Pricing & Real-Time Booking</div>", unsafe_allow_html=True)

# ==============================================================================
# AUTHENTICATION SCREEN (LOGIN / SIGNUP)
# ==============================================================================
if st.session_state["authenticated_user"] is None:
    st.markdown("### 🔑 Account Access (Login / Sign Up)")
    st.info("Please log in or create a new account as a **Driver (User)** or **Admin (Parking Operator)** to proceed.")

    auth_tab1, auth_tab2, auth_tab3 = st.tabs(["🔒 Log In", "📝 Sign Up", "⚡ Quick Demo Access"])

    # 1. LOGIN TAB
    with auth_tab1:
        st.markdown("#### Login to your Account")
        login_email = st.text_input("Email Address:", key="login_email")
        login_password = st.text_input("Password:", type="password", key="login_password")

        if st.button("Log In Now", type="primary", key="btn_login"):
            db = SessionLocal()
            try:
                user = db.query(User).filter(User.email == login_email).first()
                if user and verify_password(login_password, user.hashed_password):
                    st.session_state["authenticated_user"] = {
                        "id": user.id,
                        "name": user.name,
                        "email": user.email,
                        "role": user.role
                    }
                    st.success(f"Welcome back, {user.name}!")
                    st.rerun()
                else:
                    st.error("Invalid email or password. Please try again.")
            finally:
                db.close()

    # 2. SIGNUP TAB
    with auth_tab2:
        st.markdown("#### Create a New SPIP Account")
        signup_name = st.text_input("Full Name:", key="signup_name")
        signup_email = st.text_input("Email Address:", key="signup_email")
        signup_password = st.text_input("Password:", type="password", key="signup_password")
        signup_role = st.selectbox("Register as:", ["driver", "operator"], format_func=lambda x: "Driver / User (Book Parking)" if x == "driver" else "Admin / Parking Operator (Manage Lots)", key="signup_role")

        if st.button("Create Account", key="btn_signup"):
            if not signup_name or not signup_email or not signup_password:
                st.error("Please fill in all fields.")
            else:
                db = SessionLocal()
                try:
                    existing = db.query(User).filter(User.email == signup_email).first()
                    if existing:
                        st.error("Email is already registered. Please log in.")
                    else:
                        new_user = User(
                            name=signup_name,
                            email=signup_email,
                            hashed_password=hash_password(signup_password),
                            role=signup_role
                        )
                        db.add(new_user)
                        db.commit()
                        db.refresh(new_user)

                        st.session_state["authenticated_user"] = {
                            "id": new_user.id,
                            "name": new_user.name,
                            "email": new_user.email,
                            "role": new_user.role
                        }
                        st.success(f"Account created! Welcome, {new_user.name}.")
                        st.rerun()
                finally:
                    db.close()

    # 3. DEMO QUICK LOGIN
    with auth_tab3:
        st.markdown("#### Fast 1-Click Demo Login")
        col_demo1, col_demo2 = st.columns(2)

        with col_demo1:
            if st.button("🚗 Continue as Driver (Riya Sharma)", use_container_width=True):
                db = SessionLocal()
                try:
                    user = db.query(User).filter(User.email == "driver@spip.com").first()
                    if user:
                        st.session_state["authenticated_user"] = {"id": user.id, "name": user.name, "email": user.email, "role": user.role}
                        st.rerun()
                finally:
                    db.close()

        with col_demo2:
            if st.button("👑 Continue as Admin / Operator", use_container_width=True):
                db = SessionLocal()
                try:
                    user = db.query(User).filter(User.email == "operator@spip.com").first()
                    if user:
                        st.session_state["authenticated_user"] = {"id": user.id, "name": user.name, "email": user.email, "role": user.role}
                        st.rerun()
                finally:
                    db.close()

    st.stop()

# ==============================================================================
# LOGGED IN DASHBOARD
# ==============================================================================

user_info = st.session_state["authenticated_user"]

# User Top Status Bar & Logout
col_bar1, col_bar2, col_bar3 = st.columns([3, 1, 1])
with col_bar1:
    role_badge = "👑 ADMIN / OPERATOR" if user_info["role"] == "operator" else "🚗 DRIVER / USER"
    st.success(f"Logged in as: **{user_info['name']}** ({user_info['email']}) | Role: **{role_badge}**")
with col_bar2:
    if st.button("🔄 Trigger CV Detection Step"):
        sim_res = simulate_multi_lot_step()
        st.toast(f"CV Frame simulation processed for {len(sim_res)} lots!")
with col_bar3:
    if st.button("🚪 Logout"):
        st.session_state["authenticated_user"] = None
        st.rerun()

# Role-Based Tab Rendering
if user_info["role"] == "operator":
    tab_list = ["👑 Admin Control & Availability Update", "📊 Fleet Analytics & Forecasting", "🤖 AI Advisory Assistant"]
else:
    tab_list = ["🚗 User Place & Time Slot Booking", "📋 My Booking History", "🤖 AI Advisory Assistant"]

tabs = st.tabs(tab_list)

# ==============================================================================
# USER / DRIVER VIEW: PLACE, DATE, TIME & SLOT AVAILABILITY BOOKING
# ==============================================================================
if user_info["role"] == "driver":
    with tabs[0]:
        st.subheader("📍 Book Parking Slot by Place, Date & Time Window")

        db = SessionLocal()
        try:
            venues = db.query(Venue).all()
            venue_names = [v.name for v in venues]

            col_place, col_date, col_start, col_end = st.columns([2, 1.5, 1.5, 1.5])

            with col_place:
                selected_place = st.selectbox("Select Place / Venue:", venue_names, key="user_selected_place")

            with col_date:
                booking_date = st.date_input("Select Booking Date:", value=datetime.now().date(), min_value=datetime.now().date())

            with col_start:
                default_start = (datetime.now() + timedelta(minutes=15)).time()
                start_time_val = st.time_input("Start Time:", value=default_start)

            with col_end:
                default_end = (datetime.combine(booking_date, start_time_val) + timedelta(hours=2)).time()
                end_time_val = st.time_input("End Time:", value=default_end)

            start_datetime = datetime.combine(booking_date, start_time_val).replace(tzinfo=timezone.utc)
            end_datetime = datetime.combine(booking_date, end_time_val).replace(tzinfo=timezone.utc)

            if start_datetime >= end_datetime:
                st.error("⚠️ Start Time must be earlier than End Time.")
                st.stop()

            target_venue = next(v for v in venues if v.name == selected_place)
            target_lot = db.query(Lot).filter(Lot.venue_id == target_venue.id).first()

            if not target_lot:
                st.warning("No parking lot associated with this venue.")
                st.stop()

            slots = db.query(Slot).filter(Slot.lot_id == target_lot.id).all()

            # Dynamic Pricing & Forecast for Selected Window
            pricing_info = calculate_dynamic_price(db, target_lot.id, target_time=start_datetime)
            forecaster = ProphetLotForecaster(target_lot.id)
            fc_info = forecaster.predict_future(db, horizon_minutes=60)

            # Check Availability per slot for the chosen window
            available_slots_count = 0
            slot_availability_map = {}

            for slot in slots:
                is_avail = check_slot_availability(db, slot.id, start_datetime, end_datetime)
                slot_availability_map[slot.id] = is_avail
                if is_avail:
                    available_slots_count += 1

            duration_hours = max(0.5, (end_datetime - start_datetime).total_seconds() / 3600.0)
            calculated_price = round(target_lot.base_price_per_hour * duration_hours * pricing_info['price_multiplier'], 2)

            st.markdown("---")
            # Venue Metrics Banner
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Available Slots for Window", f"{available_slots_count} / {len(slots)} Available")
            v2.metric("Duration", f"{duration_hours:.1f} Hours", f"{start_time_val.strftime('%I:%M %p')} - {end_time_val.strftime('%I:%M %p')}")
            v3.metric("Calculated Hourly Rate", f"${pricing_info['recommended_price_per_hour']}/hr", f"Surge: {pricing_info['surge_level']}")
            v4.metric("Total Price", f"${calculated_price}")

            st.markdown(f"### 🗺️ Slot Availability Map for **{selected_place}**")
            st.caption(f"Showing slot availability for {booking_date.strftime('%b %d, %Y')} ({start_time_val.strftime('%I:%M %p')} to {end_time_val.strftime('%I:%M %p')})")

            # Slot Grid Map
            grid_cols = st.columns(5)
            selected_slot = None

            for idx, slot in enumerate(slots):
                col_idx = idx % 5
                is_free = slot_availability_map[slot.id]

                with grid_cols[col_idx]:
                    if is_free:
                        btn_label = f"🟢 Slot #{slot.slot_number}\n(AVAILABLE)"
                        if st.button(btn_label, key=f"user_slot_{slot.id}", use_container_width=True):
                            st.session_state["user_booking_slot_id"] = slot.id
                    else:
                        btn_label = f"🔴 Slot #{slot.slot_number}\n(BOOKED / OCCUPIED)"
                        st.button(btn_label, key=f"user_slot_dis_{slot.id}", disabled=True, use_container_width=True)

            # Slot Reservation Drawer / Confirmation Modal
            if "user_booking_slot_id" in st.session_state:
                res_slot_id = st.session_state["user_booking_slot_id"]
                res_slot_obj = db.query(Slot).filter(Slot.id == res_slot_id).first()

                if res_slot_obj:
                    st.markdown("---")
                    st.success(f"🎯 Selected: **Slot #{res_slot_obj.slot_number}** at {selected_place}")
                    st.markdown(f"**Reservation Window**: {booking_date.strftime('%B %d, %Y')} from **{start_time_val.strftime('%I:%M %p')}** to **{end_time_val.strftime('%I:%M %p')}** ({duration_hours:.1f} hrs)")
                    st.markdown(f"**Total Amount Charged**: **${calculated_price}**")

                    if st.button("💳 Confirm Booking & Pay Now", type="primary", key="btn_confirm_user_bkg"):
                        try:
                            bkg = create_booking_transactional(
                                db=db,
                                user_id=user_info["id"],
                                venue_id=target_venue.id,
                                lot_id=target_lot.id,
                                slot_id=res_slot_obj.id,
                                start_time=start_datetime,
                                end_time=end_datetime,
                                price_multiplier=pricing_info['price_multiplier']
                            )
                            st.balloons()
                            st.success(f"🎉 Booking Confirmed! Reference Number: **{bkg['booking_ref']}**")

                            qr_b64 = bkg["qr_code_base64"]
                            st.image(f"data:image/png;base64,{qr_b64}", width=200, caption=f"Scan QR Token at Gate: {bkg['qr_code_token']}")

                            del st.session_state["user_booking_slot_id"]
                        except Exception as ex:
                            st.error(f"Booking Error: {str(ex)}")

        finally:
            db.close()

    with tabs[1]:
        st.subheader("📋 My Active & Past Bookings")
        db = SessionLocal()
        try:
            bookings = db.query(Booking).filter(Booking.user_id == user_info["id"]).order_by(Booking.id.desc()).all()
            if bookings:
                b_records = []
                for b in bookings:
                    b_records.append({
                        "Booking Ref": b.booking_ref,
                        "Venue": b.venue.name if b.venue else "N/A",
                        "Slot Number": f"Slot #{b.slot.slot_number}" if b.slot else "N/A",
                        "Start Time": b.start_time.strftime("%b %d, %H:%M"),
                        "End Time": b.end_time.strftime("%b %d, %H:%M"),
                        "Price": f"${b.price_charged:.2f}",
                        "Status": b.status.upper()
                    })
                st.dataframe(pd.DataFrame(b_records), use_container_width=True)

                st.markdown("#### 📱 View QR Code Check-In Token")
                selected_ref = st.selectbox("Select Booking Reference:", [b.booking_ref for b in bookings])
                selected_bkg = next(b for b in bookings if b.booking_ref == selected_ref)

                from src.booking.booking_engine import generate_qr_code_base64
                qr_img = generate_qr_code_base64(selected_bkg.qr_code_token)
                st.image(f"data:image/png;base64,{qr_img}", width=180, caption=f"QR Code Token: {selected_bkg.qr_code_token}")

            else:
                st.info("No bookings recorded. Select a place and time above to reserve your parking spot!")
        finally:
            db.close()

    with tabs[2]:
        st.subheader("🤖 AI Advisory Assistant & Natural Language Booking Agent")
        st.caption("Ask questions or command the Agent in natural language (*'Book me a slot at DB City Mall for 7 PM'*).")

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Hello! I am your SPIP AI Advisory Assistant. Ask me anything about parking or say 'Book a slot at DB City Mall for 7 PM'."}
            ]

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

        if chat_in := st.chat_input("Ask a question or request a booking..."):
            st.session_state["messages"].append({"role": "user", "content": chat_in})
            st.chat_message("user").write(chat_in)

            db = SessionLocal()
            try:
                if any(w in chat_in.lower() for w in ["book", "reserve", "slot"]):
                    res = execute_agentic_booking_intent(db, user_info["id"], chat_in)
                    ans = res["message"]
                else:
                    rag_res = answer_grounded_advisory_query(db, chat_in)
                    ans = rag_res["answer"]

                st.session_state["messages"].append({"role": "assistant", "content": ans})
                st.chat_message("assistant").write(ans)
            finally:
                db.close()

# ==============================================================================
# ADMIN / OPERATOR VIEW: AVAILABILITY OVERRIDE & FLEET MANAGEMENT
# ==============================================================================
else:
    with tabs[0]:
        st.subheader("👑 Admin Control: Update Slot Availability & State Override")
        st.info("As Admin, you can manually update slot availability states, override CV detections, and edit base pricing.")

        db = SessionLocal()
        try:
            venues = db.query(Venue).all()
            selected_admin_venue = st.selectbox("Select Venue to Manage:", [v.name for v in venues], key="admin_venue")
            admin_venue_obj = next(v for v in venues if v.name == selected_admin_venue)
            admin_lot = db.query(Lot).filter(Lot.venue_id == admin_venue_obj.id).first()

            if admin_lot:
                st.markdown(f"### ⚙️ Slot Availability Manager for **{admin_lot.name}**")
                slots = db.query(Slot).filter(Slot.lot_id == admin_lot.id).all()

                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    selected_slot_num = st.selectbox("Select Slot Number:", [s.slot_number for s in slots], key="admin_slot_select")
                    target_admin_slot = next(s for s in slots if s.slot_number == selected_slot_num)
                with col_s2:
                    current_status = target_admin_slot.current_state.upper()
                    st.markdown(f"Current State: **{current_status}**")
                    new_state = st.selectbox("Update State To:", ["vacant", "reserved", "occupied", "overstayed"], index=["vacant", "reserved", "occupied", "overstayed"].index(target_admin_slot.current_state))
                with col_s3:
                    st.write("")
                    st.write("")
                    if st.button("💾 Apply State Update", type="primary", key="btn_update_slot_state"):
                        target_admin_slot.current_state = new_state
                        db.commit()
                        st.success(f"Updated Slot #{target_admin_slot.slot_number} state to **{new_state.upper()}**!")
                        st.rerun()

                st.markdown("---")
                st.markdown("### 🗺️ Live Fleet Slot Overview Map")
                grid_cols = st.columns(5)
                for idx, slot in enumerate(slots):
                    col_idx = idx % 5
                    with grid_cols[col_idx]:
                        badge = "🟢 VACANT" if slot.current_state == "vacant" else ("🟡 RESERVED" if slot.current_state == "reserved" else ("🔴 OCCUPIED" if slot.current_state == "occupied" else "🟣 OVERSTAYED"))
                        st.markdown(f"**Slot #{slot.slot_number}**\n{badge}")

                st.markdown("---")
                st.markdown("### 💵 Base Hourly Rate Control")
                col_p1, col_p2 = st.columns([2, 1])
                with col_p1:
                    new_base_price = st.number_input("Base Hourly Price ($):", min_value=1.0, max_value=50.0, value=float(admin_lot.base_price_per_hour), step=0.5)
                with col_p2:
                    st.write("")
                    st.write("")
                    if st.button("Update Pricing", key="btn_update_pricing"):
                        admin_lot.base_price_per_hour = new_base_price
                        db.commit()
                        st.success(f"Base price updated to **${new_base_price}/hr**!")

        finally:
            db.close()

    with tabs[1]:
        st.subheader("📊 Fleet Analytics & Predictive Forecasting")
        db = SessionLocal()
        try:
            all_lots = db.query(Lot).all()
            selected_op_lot_name = st.selectbox("Select Lot for Analytics:", [l.name for l in all_lots])
            lot_obj = next(l for l in all_lots if l.name == selected_op_lot_name)

            events = db.query(OccupancyEvent).filter(OccupancyEvent.lot_id == lot_obj.id).order_by(OccupancyEvent.timestamp.asc()).all()
            if events:
                df_ev = pd.DataFrame([{"timestamp": e.timestamp, "is_occupied": e.is_occupied} for e in events])
                df_ev['timestamp'] = pd.to_datetime(df_ev['timestamp'])
                df_agg = df_ev.groupby(pd.Grouper(key='timestamp', freq='1h')).agg(occupied=('is_occupied', 'sum')).reset_index()
                df_agg['occupancy_pct'] = (df_agg['occupied'] / lot_obj.total_slots) * 100.0

                fig = px.line(df_agg, x="timestamp", y="occupancy_pct", title=f"Hourly Occupancy % for {lot_obj.name}")
                fig.add_hline(y=75, line_dash="dash", line_color="orange", annotation_text="Surge Threshold (75%)")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📝 Generate Automated Operator Report")
            if st.button("📄 Generate Full Executive Report"):
                rep = generate_operator_summary_report(db)
                st.markdown(rep["report_markdown"])
        finally:
            db.close()

    with tabs[2]:
        st.subheader("🤖 AI Advisory Assistant")
        if "admin_messages" not in st.session_state:
            st.session_state["admin_messages"] = [{"role": "assistant", "content": "Admin Advisory Online. Ask any query regarding fleet policy or utilization."}]

        for msg in st.session_state["admin_messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

        if admin_in := st.chat_input("Type your query..."):
            st.session_state["admin_messages"].append({"role": "user", "content": admin_in})
            st.chat_message("user").write(admin_in)

            db = SessionLocal()
            try:
                ans_res = answer_grounded_advisory_query(db, admin_in)
                st.session_state["admin_messages"].append({"role": "assistant", "content": ans_res["answer"]})
                st.chat_message("assistant").write(ans_res["answer"])
            finally:
                db.close()
