from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DriftMetric(BaseModel):
    id: int
    batch_id: int
    timestamp: datetime
    feature_name: str
    drift_score: Optional[float]
    is_drifted: bool
    alert_message: Optional[str]
    metric_type: str

    class Config:
        orm_mode = True