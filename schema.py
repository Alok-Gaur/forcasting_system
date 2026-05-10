from pydantic import BaseModel, Field
from typing import Literal

class PredictionInput(BaseModel):
    state: Literal[
        'Alabama', 'Arizona', 'Arkansas', 'California', 'Colorado',
        'Connecticut', 'Florida', 'Georgia', 'Illinois', 'Indiana', 'Iowa',
        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
        'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
        'Missouri', 'Nebraska', 'Nevada', 'New Hampshire', 'New Mexico',
        'New York', 'North Carolina', 'Ohio', 'Oklahoma', 'Oregon',
        'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
        'West Virginia', 'Wisconsin', 'Wyoming'
    ] = Field(default="New York")
    weeks: int = Field(gt=0, le=80)
