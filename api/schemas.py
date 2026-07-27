from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

class PredictionResponse(BaseModel):
    filename: str
    predicted: bool = Field(..., description='Whether pneumonia was detected')
    confidence: float = Field(..., description='Highest confidence score across all detections')
    n_detections: int = Field(..., description='Number of bounding boxes detected')
    boxes: list[BoundingBox] = Field(default_factory=list)

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str