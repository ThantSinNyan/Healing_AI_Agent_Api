from pydantic import BaseModel

class HealingRequest(BaseModel):
    birth_date: str
    time: str
    birth_place: str
    language: str

class HealingResponse(BaseModel):
    mainTitle: str
    description: str
    CoreWoundsAndEmotionalThemes: list[str]
    PatternsAndStruggles: list[str]
    HealingAndTransformation: list[str]
    SpiritualWisdomAndGifts: list[str]
    woundPoints: list[str]
    PatternsConnectedToThisWound: list[str]
    HealingBenefits: list[str]

class ChironAnalysisResponse(BaseModel):
    placement: str
    coreWoundedThemes: str
    summaryOverview: list[str]
    woundedKeywords: list[str]
    healingKeywords: list[str]
    primaryChallenges: list[str]
    pathToHealing: list[str]

class ChironRequest(BaseModel):
    sign: str
    house: str

class ChironResponse(BaseModel):
    answer: str