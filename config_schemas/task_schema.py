from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass

from config_schemas.task import (
    loss_function_schema,
    model_schema,
    optimizer_schema
)

@dataclass
class TaskConfig:
    _target_: str = MISSING
    optimizer: optimizer_schema.OptimizerConfig = MISSING
    model: model_schema.ModelConfig = MISSING
    loss_function: loss_function_schema.LossFunctionConfig = MISSING
    

@dataclass
class MNISTClassificationTaskConfig(TaskConfig):
    _target_: str = "tasks.MNISTClassification"
    
    
@dataclass
class CIFAR10ClassificationTaskConfig(TaskConfig):
    _target_: str = "tasks.CIFAR10Classification"
    model: model_schema.ModelConfig = MISSING
    
    
    
def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="task", 
        name="mnist_classfication_task_schema",
        node=MNISTClassificationTaskConfig
    )
    cs.store(
        group="task",
        name="cifar10_classification_task_schema",
        node=CIFAR10ClassificationTaskConfig
    )
    
    optimizer_schema.setup_config()
    model_schema.setup_config()
    loss_function_schema.setup_config()
    
    