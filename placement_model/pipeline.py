import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from placement_model.config.core import config
from placement_model.processing.features import Mapper
placement_pipe=Pipeline([
    
     ##==========Mapper======##

     ("map_sex", Mapper(config.model_config.gender_var, config.model_config.gender_mappings)),
     ("map_ssc_b", Mapper(config.model_config.ssc_b_var, config.model_config.ssc_b_mappings )),
     ("map_hsc_b", Mapper(config.model_config.hsc_b_var, config.model_config.hsc_b_mappings)),
     ("map_hsc_s", Mapper(config.model_config.hsc_s_var, config.model_config.hsc_s_mappings)),
     ("map_degree_t", Mapper(config.model_config.degree_t_var, config.model_config.degree_t_mappings)),
     ("map_workex", Mapper(config.model_config.workex_var, config.model_config.workex_mappings)),
     ("map_specialisation", Mapper(config.model_config.specialisation_var, config.model_config.specialisation_mappings)),
     # scale
     ("scaler", StandardScaler()),
     ('model_rf', RandomForestClassifier(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth,
                                      random_state=config.model_config.random_state))
          
     ])