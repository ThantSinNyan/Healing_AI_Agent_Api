# app/routers/healing_router.py

from fastapi import APIRouter
from app.models.healing_model import HealingRequest, HealingResponse
from app.services.healing_service import generate_healing

router = APIRouter()

@router.post("/generate", response_model=HealingResponse)
def generate_healing_endpoint(request: HealingRequest):
    return generate_healing(request.zodiac_sign, request.house, request.language)
