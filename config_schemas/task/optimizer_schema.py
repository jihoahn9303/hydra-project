from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass


@dataclass 
class OptimizerConfig:
    _target_: str = MISSING
    _partial_: bool = True
    lr: float = 0.0001
    weight_decay: float = 0.0


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    _target_: str = "torch.optim.Adam"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    
    
@dataclass
class SGDOptimizerConfig(OptimizerConfig):
    _target_: str = "torch.optim.SGD"
    

def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="task/optimizer", 
        name="adam_optimizer_schema", 
        node=AdamOptimizerConfig
    )
    cs.store(
        group="task/optimizer",
        name="sgd_optimizer_schema",
        node=SGDOptimizerConfig
    )