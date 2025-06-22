from pydantic import BaseModel

class HealingRequest(BaseModel):
    zodiac_sign: str
    house: str
    language: str

class HealingResponse(BaseModel):
    description: str
    woundPoints: list[str]