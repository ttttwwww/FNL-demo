import abc
import time
from . import logger
import numpy as np


class BaseTrainer(abc.ABC):
    """
    Trainer for simulator. It controls the training process of the model.
    """
    def __init__(self,ctx:dict):
        """
        Initialize the trainer with the context.Including the basic element for a training.
        Args:
            ctx: context including:
                optimizer: The optimizer to be used.
                model: The model to be trained.
                criterion: The loss function to be used.

        """
        self.model = ctx.get("model")
        self.optimizer = ctx.get("optimizer")
        self.criterion = ctx.get("criterion")
        self.epochs = ctx.get("epochs")

    def train(self,**kwargs)->dict:
        """
        To be implemented in subclass.
        Returns:
            result: Dict with training results.Including:
                train_acc, train_loss, test_acc, test_loss
        """
        raise NotImplementedError
#TODO finish the trainer
class DefaultTrainer(BaseTrainer):
    def __init__(self,ctx:dict):
        super().__init__(ctx)

    def train(self,train_loader)->dict:
        """
        Perform training as the config files.
        Results will output as dict.
        Including:
            train_acc,train_loss,test_acc,test_loss
        """
        train_acc = np.empty(0)
        train_loss = np.empty(0)
        test_acc = np.empty(0)
        test_loss = np.empty(0)
        best_acc = 0

        start_time = time.time()
        logger.info("Start training...")
        for epoch in range(self.epochs):
            self.model.train()
            num_sample, num_acc, num_loss = self._epoch_run(self.train_loader, train=True)
            epoch_acc = num_acc / num_sample
            epoch_loss = num_loss / num_sample
            train_acc = np.append(train_acc, epoch_acc)
            train_loss = np.append(train_loss, epoch_loss)
            logger.info(f"Epoch {epoch + 1}/{self.epochs}: train:acc: {epoch_acc:.4f},train_loss: {epoch_loss:.4f} ")

            self.model.eval()
            num_sample, num_acc, num_loss = self._epoch_run(self.test_loader, train=False)
            epoch_acc = num_acc / num_sample
            epoch_loss = num_loss / num_sample
            test_acc = np.append(test_acc, epoch_acc)
            test_loss = np.append(test_loss, epoch_loss)
            logger.info(f"Epoch {epoch + 1}/{self.epochs}: test:acc: {epoch_acc:.4f},test_loss: {epoch_loss:.4f} ")

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                logger.info(f"New best val acc : {best_acc:.4f} in epoch {epoch + 1}")
                self.save_model()
        end_time = time.time()
        logger.info(f"Finish training... cost time: {end_time - start_time:.2f}s")
        logger.info(f"Best val acc : {best_acc:.4f}")
        result = {"train_acc": train_acc, "train_loss": train_loss, "test_acc": test_acc, "test_loss": test_loss}
        return result