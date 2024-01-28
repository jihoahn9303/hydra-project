from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class TrainerConfig:
    _target_: str = "pytorch_lightning.Trainer"
    max_epochs: int = 10
    precision: int = 16
    log_every_n_steps: int = 10
    accelerator: str = MISSING
    devices: str = "auto"
    
    
def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="trainer", 
        name="trainer_schema", 
        node=TrainerConfig
    )
    