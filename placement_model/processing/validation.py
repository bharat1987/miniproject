import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from placement_model.config.core import config
from placement_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):

    sl_no: Optional[int]
    gender: Optional[str]
    ssc_p: Optional[float]
    ssc_b: Optional[str]
    hsc_p: Optional[float]
    hsc_b: Optional[str]
    hsc_s: Optional[str]
    degree_p: Optional[float]
    degree_t: Optional[str]
    workex: Optional[str]
    etest_p: Optional[float]
    specialisation: Optional[str]
    mba_p: Optional[float]
    status: Optional[str]
    salary: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]