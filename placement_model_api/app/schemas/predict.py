from typing import Any, List, Optional

from numpy import ndarray

from pydantic import BaseModel
from placement_model.processing.validation import DataInputSchema

class PredictionResults(BaseModel):
    predictions: Optional[list[str]]
    errors: Optional[Any]
    version: str
    
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "sl_no":100,
                        "gender":"M",
                        "ssc_p":62.00,
                        "ssc_b":"Central",
                        "hsc_p":58.00,
                        "hsc_b":"Others",
                        "hsc_s":"Science",
                        "degree_p": 53.00,
                        "degree_t": "Comm&Mgmt",
                        "workex": "No",
                        "etest_p":89,
                        "specialisation": "Mkt&HR",
                        "mba_p":60.22
                    },
                    {
                        "sl_no":101,
                        "gender":"F",
                        "ssc_p":83.96,
                        "ssc_b":"Others",
                        "hsc_p":53.00,
                        "hsc_b":"Others",
                        "hsc_s":"Science",
                        "degree_p": 91.00,
                        "degree_t": "Sci&Tech",
                        "workex": "No",
                        "etest_p":59.32,
                        "specialisation": "Mkt&HR",
                        "mba_p":69.71
                    }
                ]
            }
        }
