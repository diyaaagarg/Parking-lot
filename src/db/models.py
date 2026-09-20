from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from src.db.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="driver")  # driver, operator
    created_at = Column(DateTime, default=utc_now)

    bookings = relationship("Booking", back_populates="user")

class Venue(Base):
    __tablename__ = "venues"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(150), nullable=False)
    address = Column(String(255), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    category = Column(String(50), nullable=False)  # mall, theater, downtown, airport
    total_capacity = Column(Integer, default=0)

    lots = relationship("Lot", back_populates="venue")
    bookings = relationship("Booking", back_populates="venue")

class Lot(Base):
    __tablename__ = "lots"

    id = Column(Integer, primary_key=True, index=True)
    venue_id = Column(Integer, ForeignKey("venues.id"), nullable=False)
    name = Column(String(100), nullable=False)
    total_slots = Column(Integer, default=0)
    base_price_per_hour = Column(Float, default=5.0)

    venue = relationship("Venue", back_populates="lots")
    slots = relationship("Slot", back_populates="lot")
    occupancy_events = relationship("OccupancyEvent", back_populates="lot")
    bookings = relationship("Booking", back_populates="lot")

class Slot(Base):
    __tablename__ = "slots"

    id = Column(Integer, primary_key=True, index=True)
    lot_id = Column(Integer, ForeignKey("lots.id"), nullable=False)
    slot_number = Column(Integer, nullable=False)
    polygon_coordinates = Column(Text, nullable=False)  # JSON representation of 4 points [[x1,y1],[x2,y2]...]
    current_state = Column(String(20), default="vacant")  # vacant, reserved, occupied, overstayed

    lot = relationship("Lot", back_populates="slots")
    occupancy_events = relationship("OccupancyEvent", back_populates="slot")
    bookings = relationship("Booking", back_populates="slot")

class OccupancyEvent(Base):
    __tablename__ = "occupancy_events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=utc_now, index=True)
    lot_id = Column(Integer, ForeignKey("lots.id"), nullable=False)
    slot_id = Column(Integer, ForeignKey("slots.id"), nullable=False)
    is_occupied = Column(Integer, nullable=False)  # 1 occupied, 0 vacant

    lot = relationship("Lot", back_populates="occupancy_events")
    slot = relationship("Slot", back_populates="occupancy_events")

class Booking(Base):
    __tablename__ = "bookings"

    id = Column(Integer, primary_key=True, index=True)
    booking_ref = Column(String(50), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    venue_id = Column(Integer, ForeignKey("venues.id"), nullable=False)
    lot_id = Column(Integer, ForeignKey("lots.id"), nullable=False)
    slot_id = Column(Integer, ForeignKey("slots.id"), nullable=False)
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=False, index=True)
    status = Column(String(30), default="confirmed")  # confirmed, completed, cancelled, overstayed
    price_charged = Column(Float, nullable=False)
    qr_code_token = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=utc_now)

    user = relationship("User", back_populates="bookings")
    venue = relationship("Venue", back_populates="bookings")
    lot = relationship("Lot", back_populates="bookings")
    slot = relationship("Slot", back_populates="bookings")
    transaction = relationship("Transaction", back_populates="booking", uselist=False)

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    booking_id = Column(Integer, ForeignKey("bookings.id"), nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String(20), default="success")  # success, refunded, failed
    payment_ref = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=utc_now)

    booking = relationship("Booking", back_populates="transaction")

class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user_query = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now)

class KnowledgeDoc(Base):
    __tablename__ = "knowledge_docs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False)  # policy, venue_info, pricing_rule
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now)
