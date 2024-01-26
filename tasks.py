from typing import Callable, Iterator

# import pytorch_lightning as pl
from torch import Tensor, optim
from torchmetrics import Accuracy
from torch import nn
from pytorch_lightning import LightningModule


PARTIAL_OPTIMIZER_TYPE = Callable[[Iterator[nn.Parameter]], optim.Optimizer]


class TrainingTask(LightningModule):
    def __init__(self, optimizer: PARTIAL_OPTIMIZER_TYPE) -> None:
        super().__init__()
        self.optimizer = optimizer

    def configure_optimizers(self) -> optim.Optimizer:
        return self.optimizer(params=self.parameters())


class MNISTClassification(TrainingTask):
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: PARTIAL_OPTIMIZER_TYPE, 
        loss_function: nn.Module
    ) -> None:
        super().__init__(optimizer)

        self.model = model
        self.loss_function = loss_function

        num_classes = 10
        
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.validaiton_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, **kwargs) -> Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.loss_function(logits, labels)
        self.train_accuracy(logits, labels)
        self.log(
            name="train_loss", 
            value=loss, 
            on_step=True, 
            on_epoch=True
        )
        self.log(
            name="train_accuracy", 
            value=self.train_accuracy, 
            on_step=False, 
            on_epoch=True
        )
        
        return loss

    def validation_step(self, batch, **kwargs):
        images, labels = batch
        preds = self(images)
        self.validaiton_accuracy(preds, labels)
        self.log(
            name="validation_accuracy", 
            value=self.validaiton_accuracy, 
            on_step=False, 
            on_epoch=True
        )

    def test_step(self, batch, **kwargs):
        images, labels = batch
        preds = self(images)
        self.test_accuracy(preds, labels)
        self.log(
            name="test_accuracy", 
            value=self.test_accuracy, 
            on_step=False, 
            on_epoch=True
        )


class CIFAR10Classification(TrainingTask):
    """
    Hint: It is going to be very similar to MNISTClassification
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: PARTIAL_OPTIMIZER_TYPE,
        loss_function: nn.Module
    ) -> None:
        super().__init__(optimizer)
        
        self.model = model
        self.loss_function = loss_function
        
        num_classes = 10
        
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.validation_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)    
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def training_step(self, batch: list[Tensor], _batch_idx: int):
        images, labels = batch
        logits = self(images)
        loss = self.loss_function(logits, labels)
        self.train_accuracy(logits, labels)
        self.log(
            name="train_loss", 
            value=loss, 
            on_step=True, 
            on_epoch=True
        )
        self.log(
            name="train_accuracy", 
            value=self.train_accuracy, 
            on_step=False, 
            on_epoch=True
        )
        
        return loss
    
    def validation_step(self, batch: list[Tensor], _batch_idx: int):
        images, labels = batch
        preds = self(images)
        self.validation_accuracy(preds, labels)
        self.log(
            name="validation_accuracy", 
            value=self.validation_accuracy, 
            on_step=False, 
            on_epoch=True
        )

    def test_step(self, batch: list[Tensor], _batch_idx: int):
        images, labels = batch
        preds = self(images)
        self.test_accuracy(preds, labels)
        self.log(
            name="test_accuracy", 
            value=self.test_accuracy, 
            on_step=False, 
            on_epoch=True
        )
