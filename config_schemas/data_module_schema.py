from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass


@dataclass
class DataModuleConfig:
    _target_: str = MISSING
    pin_memory: bool = True
    drop_last: bool = True
    batch_size: int = MISSING
    num_workers: int = MISSING
    data_dir: str = MISSING


@dataclass
class MNISTDataModuleConfig(DataModuleConfig):
    _target_: str = "data_modules.MNISTDataModule"
    
    
@dataclass
class CIFAR10DataModuleConfig(DataModuleConfig):
    _target_: str = "data_modules.CIFAR10DataModule"

    
def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="data_module", 
        name="mnist_data_module_schema", 
        node=MNISTDataModuleConfig
    )
    cs.store(
        group="data_module",
        name="cifar10_data_module_schema",
        node=CIFAR10DataModuleConfig
    )
    