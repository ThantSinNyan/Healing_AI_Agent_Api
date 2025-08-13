# app/models/qa_models.py
from pydantic import BaseModel

class ChironRequest(BaseModel):
    sign: str
    house: str

class ChironResponse(BaseModel):
    answer: str

class TestResult(BaseModel):
    query: str
    ground_truth: str
    answer: str
    raw_score: float
    score: float
