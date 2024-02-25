
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from placement_model.config.core import config
from placement_model.processing.features import Mapper


def test_specialisation_transformer(sample_input_data):
    # Given
    transformer = Mapper(
        variables=config.model_config.specialisation_var, mappings=config.model_config.specialisation_mappings # cabin
    )
    
    print(sample_input_data.loc[183,'specialisation'])
    assert isinstance(sample_input_data.loc[183,'specialisation'], str)

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert subject.loc[183,'specialisation'] == 0