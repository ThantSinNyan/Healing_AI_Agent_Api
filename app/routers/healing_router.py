# app/routers/healing_router.py

from fastapi import APIRouter, HTTPException
from app.models.healing_model import HealingRequest, HealingResponse, ChironRequest, ChironResponse, ChironAnalysisResponse
from app.services.healing_service import generate_healing
from app.services import qa_accuracy_tester 

router = APIRouter()

@router.post("/generate", response_model=HealingResponse)
def generate_healing_endpoint(request: HealingRequest):
    return generate_healing(request.birth_date, request.time, request.birth_place,request.language)


@router.post("/chiron_analysis", response_model=ChironAnalysisResponse)
async def chiron_analysis(request: ChironRequest):
    try:
        answer = qa_accuracy_tester.get_combined_chiron_answer(
            sign=request.sign,
            house=request.house,
            k=5
        )
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "ok"}
