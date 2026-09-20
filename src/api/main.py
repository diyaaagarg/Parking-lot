from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import API_HOST, API_PORT
from src.db.database import engine, Base
from src.api.auth_router import router as auth_router
from src.api.venues_router import router as venues_router
from src.api.booking_router import router as booking_router
from src.api.forecast_router import router as forecast_router
from src.api.rag_router import router as rag_router

# Create DB tables if not existing
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart Parking Intelligence Platform (SPIP v2) API",
    description="Multi-Lot Occupancy Detection, Forecasting, Dynamic Pricing & AI-Powered Advisory System API",
    version="2.0.0"
)

# Enable CORS for Streamlit Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(auth_router)
app.include_router(venues_router)
app.include_router(booking_router)
app.include_router(forecast_router)
app.include_router(rag_router)

@app.get("/")
def root():
    return {
        "status": "online",
        "system": "Smart Parking Intelligence Platform (SPIP v2)",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=True)
