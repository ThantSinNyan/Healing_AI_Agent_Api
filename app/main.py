from fastapi import FastAPI
from app.routers import healing_router

app = FastAPI(title="Healing Journey API")

app.include_router(healing_router.router, prefix="/healing")