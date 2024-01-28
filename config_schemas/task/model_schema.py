from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass

from config_schemas.task.model import (
    adapter_schema,
    backbone_schema,
    head_schema
)

@dataclass
class ModelConfig:
    _target_: str = MISSING
    backbone: backbone_schema.BackboneConfig = MISSING
    adapter: adapter_schema.AdapterConfig = MISSING
    head: head_schema.HeadConfig = MISSING
    

@dataclass
class SimpleModelConfig(ModelConfig):
    _target_: str = "models.SimpleModel"
      

@dataclass
class CIFAR10ModelConfig(ModelConfig):
    _target_: str = "models.CIFAR10Model"
    
    

def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="task/model", 
        name="simple_model_schema", 
        node=SimpleModelConfig
    )
    cs.store(
        group="task/model",
        name="cifar10_model_schema",
        node=CIFAR10ModelConfig
    )
    
    
    backbone_schema.setup_config()
    adapter_schema.setup_config()
    head_schema.setup_config()    