from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass

from config_schemas import (
    data_module_schema,
    trainer_schema,
    task_schema
) 


@dataclass
class Config:
    task: task_schema.TaskConfig = MISSING
    data_module: data_module_schema.DataModuleConfig = MISSING
    trainer: trainer_schema.TrainerConfig = MISSING
    

def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
    
    task_schema.setup_config()
    data_module_schema.setup_config()
    trainer_schema.setup_config()
    