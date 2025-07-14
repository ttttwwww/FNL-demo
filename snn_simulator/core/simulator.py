"""
The core of the SNN simulation. Managing the parameters and dataflow in network training.
Different simulators are created to adapt different SNN.Such as timestep based network or precise time based network
"""
import copy
import importlib
import os
import time
from collections import defaultdict
from datetime import datetime

import pandas as pd
import torchvision
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torchviz import make_dot
from tqdm import tqdm
import seaborn as sns

import snn_simulator.utils.loss_utils
from . import logger
from abc import ABC, abstractmethod
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from typing import cast, Sized, Optional
from snn_simulator.utils.config_utils import merge_config, get_object_from_str
from snn_simulator.config.default_config import DEFAULT_CONFIG
from snn_simulator.models.base import BaseTimeStepNetwork
from snn_simulator.utils.hook import TensorCollector


# TODO Modularize internal functionality
class BaseSimulator(ABC):
    """
    Base Simulator. Define the interface for all simulators.
    Simulator help to control the data flaw during the simulation, including
        Load data from dataset
        Create training environment such as optimizer,criterion,scheduler,...
        Perform forward and backward process
        Saving training record and the best config the net weights.
    """

    def __init__(self, config: dict, train_dataset: Dataset, test_dataset: Dataset, val_dataset: Dataset = None,
                 debug: bool = False):
        """
        Base Simulator only have model and dataset.The reset parameters are supposed to be loaded in subclasses
        Args:
            train_dataset: The dataset to be used in training. Should be encapsulated in Dataset class.
            test_dataset: The dataset for testing. Should be encapsulated in Dataset class.
            config: The config for the simulation
            debug: Save the tensors in the simulation for debugging purposes if is True.
        """
        self.tensor_collector = TensorCollector()
        self.model: Optional[nn.Module] = None
        self.acc_computer = None
        self.save_root = None
        self.epochs = None
        self.criterion = None
        self.scheduler = None
        self.optimizer = None

        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.debug = debug
        self.device = config["common"]["device"]  # TODO find a better way to handle this
        self.simulator_init()

    @abstractmethod
    def run(self):
        """
        The concrete implementation of the simulation.Should be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def simulator_init(self):
        """
        Load parameters from config file and prepare training component.
        Should be implemented in subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def _plot_confusion_matrix(save_root: str, confusion_matrix: np.ndarray) -> None:
        """
        Plot confusion matrix after training.
        Args:
            save_root: Root directory to save the figure.
            confusion_matrix: Confusion matrix after training.
        """
        save_path = os.path.join(save_root, "data/confusion_matrix.csv")
        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savetxt(save_path, confusion_matrix, delimiter=",")
        save_path = os.path.join(save_root, "fig/confusion_matrix.png")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sns.heatmap(confusion_matrix, annot=True)
        plt.title("Confusion Matrix")
        plt.savefig(save_path)
        plt.show()
        plt.clf()
        logger.info(f"Confusion matrix saved to {save_path}")

    @staticmethod
    def _plot_computation_graph(save_root, model: nn.Module, loss: torch.Tensor) -> None:
        """
        Save the compute map of model in simulator.
        Args:
            loss:  Loss of the model output.
            save_root: Root dir for saving.
        """
        dot = make_dot(loss, params=dict(model.named_parameters()))
        save_path = os.path.join(save_root, "fig/computation_graph")
        dot.render(save_path, format='png')
        logger.info(f" Computation graph saved to {save_path}")

    @staticmethod
    def _plot_acc(save_root, train_acc, train_loss, test_acc, test_loss, *args, **kwargs) -> None:
        """
        Plot accuracy and loss curves after training.
        Args:
            save_root: Root dir for saving.
            train_acc: Train accuracy after training.
            train_loss: Train loss after training.
            test_acc: Test accuracy after training.
            test_loss: Test loss after training.
        """
        save_path = os.path.join(save_root, "fig/")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.plot(train_acc, label="train acc")
        plt.title("Train Accuracy")
        plt.savefig(os.path.join(save_root, "fig/train_acc.png"))
        plt.show()
        plt.clf()
        plt.plot(train_loss, label="train loss")
        plt.title("Train Loss")
        plt.savefig(os.path.join(save_root, "fig/train_loss.png"))
        plt.show()
        plt.clf()

        plt.plot(test_acc, label="test acc")
        plt.title("Test Accuracy")
        plt.savefig(os.path.join(save_root, "fig/test_acc.png"))
        plt.show()
        plt.clf()
        plt.plot(test_loss, label="test loss")
        plt.title("Test Loss")
        plt.savefig(os.path.join(save_root, "fig/test_loss.png"))
        plt.show()
        plt.clf()

        save_path = os.path.join(save_root, "data/")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savetxt(os.path.join(save_path, "train_acc.csv"), train_acc, delimiter=",")
        np.savetxt(os.path.join(save_path, "train_loss.csv"), train_loss, delimiter=",")
        np.savetxt(os.path.join(save_path, "test_acc.csv"), test_acc, delimiter=",")
        np.savetxt(os.path.join(save_path, "test_loss.csv"), test_loss, delimiter=",")

    def get_model(self) -> nn.Module:
        """
        Get the model in simulator.
        Returns:
            model: Network created by config
        """
        return self.model

    def model_init(self) -> None:
        """
        Initialize the model in simulator.
        Creating new model and decide if loading ckpt
        """
        self.model = self._creat_model()
        if "load_path" in self.config["model"] and self.config["model"]["load_path"] is not None:
            load_path = self.config["model"]["load_path"]
            self.model.load_state_dict(torch.load(load_path))

        # register grad hook if in debug mode
        if self.debug:
            logger.info(f"register hook to collect grads in model {self.model}")
            for name, param in self.model.named_parameters():
                param.register_hook(lambda grad, n=name: self.tensor_collector.store_grad_param(n, grad))
            for name, param in self.model.named_modules():
                if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
                    def backward_hook(module, grad_input, grad_output, layer_name=name):
                        grad_name = f"{layer_name}_grad_input"
                        if grad_input[0] is not None:
                            self.tensor_collector.store_grad_param(grad_name, grad_input[0])
                        grad_name = f"{layer_name}_grad_output"
                        if grad_output[0] is not None:
                            self.tensor_collector.store_grad_param(grad_name, grad_output[0])

                    param.register_full_backward_hook(backward_hook)

    def _creat_model(self) -> nn.Module:
        """
        Create a new model with params.
        Returns:
            model: Network created by config, subclass of nn.Module.
        """
        # default_config = DEFAULT_CONFIG["model"]
        # config = merge_config(default_config, self.config["model"])
        config = self.config["model"]
        model_cls = get_object_from_str(config["type"])
        model_params = config["model_params"]
        model_params["debug"] = self.debug
        model_params["tensor_collector"] = self.tensor_collector
        model = None
        try:
            model = model_cls(**model_params)
            model = model.to(self.device)
        except Exception as e:
            logger.error(e)
            logger.error(f"Model params are not compatible for model_cls: {model_cls}")
            if self.debug:
                raise e
        return model

    def environment_init(self) -> None:
        """
        Initialize the environment.Including the optimizer ,scheduler and criterion.
        """
        default_config = DEFAULT_CONFIG["common"].copy()
        config = merge_config(default_config, self.config["common"])
        self.config["common"] = config.copy()
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_root = os.path.join(config["save_root"], time_str)
        self.device = config['device']
        self.epochs = config['epochs']

        try:
            self.acc_computer = get_object_from_str(config['acc_computer'])
        except Exception as e:
            logger.error(e)
            logger.error(f"The acc_computer are not compatible")
            if self.debug:
                raise e
        try:  # TODO creation factory method to create optimizer
            optimizer_cls = get_object_from_str(config['optimizer'].pop("type"))
            self.optimizer = optimizer_cls(self.model.parameters(), **config['optimizer'])
        except Exception as e:
            logger.error(e)
            logger.error(f"The optimizer params are not compatible")
            if self.debug:
                raise e

        if config["scheduler"] is not None:
            try:
                scheduler_cls = get_object_from_str(config['scheduler'].pop("type"))
                self.scheduler = scheduler_cls(self.optimizer, **config['scheduler'])
            except Exception as e:
                logger.error(e)
                logger.error(f"The scheduler params are not compatible")
                if self.debug:
                    raise e
        try:
            criterion_cls = get_object_from_str(config["criterion"]["type"])
            params = config["criterion"].get("params", {})
            if params is not None:
                self.criterion = criterion_cls(**params)
            else:
                self.criterion = criterion_cls()
        except Exception as e:
            logger.error(e)
            logger.error(f"The criterion params are not compatible")
            if self.debug:
                raise e

    @abstractmethod
    def dataloader_init(self, *args, **kwargs) -> None:
        """
        Initialize the dataloader.To convert raw_dataset into dataloader.
        Should be implemented in subclasses.
        """
        raise NotImplementedError


class TimeStepBasedSimulator(BaseSimulator):
    """
    TimeStepBasedSimulator is for timestep style simulation.
    The neuron and network will run in discrete time step mode.
    """

    def __init__(self, config: dict, train_dataset: Dataset, test_dataset: Dataset, val_dataset: Dataset = None,
                 debug: bool = False):
        """
        Args:
            config: See the description in BaseSimulator.
            train_dataset: See the description in BaseSimulator.
            test_dataset: See the description in BaseSimulator.
            debug: See the description in BaseSimulator.
        """
        self.max_spike_time = config["globals"]["max_spike_time"]
        super().__init__(config, train_dataset, test_dataset, val_dataset, debug)

    def simulator_init(self) -> None:
        """
        Initialize the simulator
        """
        self.model_init()
        self.dataloader_init()
        self.environment_init()

    def dataloader_init(self):
        """
        Make DataLoader from dataset and config parameters.
        """
        default_config = DEFAULT_CONFIG["dataloader"].copy()
        config = merge_config(default_config, self.config["dataloader"])
        self.config["dataloader"] = config

        batch_size = config['batch_size']
        train_shuffle = config['train_shuffle']
        test_shuffle = config['test_shuffle']
        val_shuffle = config['val_shuffle']
        num_workers = config['num_workers']
        val_dataset = None
        # TODO remove val_dataset temporally,should be implemented in future
        # if "val_size" in config:
        #     val_size = config["val_size"]
        #     train_dataset, test_dataset, val_dataset = random_split(
        #         self.dataset,
        #         [int(dataset_size * train_size), int(dataset_size * test_size), int(dataset_size * val_size)])
        # else:
        #     train_dataset, test_dataset = random_split(self.dataset,
        #                                                [int(dataset_size * train_size), int(dataset_size * test_size)])
        #     val_dataset = None

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=train_shuffle,
                                       num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=test_shuffle,
                                      num_workers=num_workers)
        # if val_dataset is not None:
        #     self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=val_shuffle,
        #                                  num_workers=num_workers)
        if self.val_loader is not None:
            self.val_loader = DataLoader(self.val_loader, batch_size=batch_size, shuffle=val_shuffle, )

    def run(self) -> None:
        """
        Control the overall simulation.
        """
        result = self.train()
        self._generate_report(result)

    def train(self) -> dict:
        """
        Perform training as the config files.
        Results will output as dict.
        Including:
            train_acc,train_loss,test_acc,test_loss
        Returns:
            result: Dict with training results.
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

    def _generate_report(self, train_result):
        """
        Generate a report describing the training and validation accuracy.
        This function only implement basic function. Custom function should be implemented in subclass.
        Basic report including:
            Train accuracy and loss
            Test accuracy and loss
            Confusion matrix
            Computation graph
        """
        logger.info("Generate report------------------------------------------------------------------")
        self._plot_run(self.test_loader)
        self._plot_acc(save_root=self.save_root, **train_result)
        self.save_config()

    def _epoch_run(self, dataloader: torch.utils.data.DataLoader, train: bool) -> [int, int, float]:
        """
        Execute one epoch of training or testing.Return the sample number, acc number and loss array
        Args:
            dataloader: Prepared dataloader.
        Returns:
            num_sample: Number of sample in epoch
            num_acc: Correct number of sample in epoch
            num_loss: Sum loss of each sample.
        """
        num_sample, num_acc, num_loss = 0, 0, 0.
        pbar = tqdm(dataloader)
        for img, label in pbar:
            if isinstance(img, torch.Tensor):
                img = img.to(self.device)
            elif isinstance(img, list):
                img = [i.to(self.device) for i in img]
            else:
                raise TypeError(
                    f"Unsupported data type {type(img)} in dataloader, should be Tensor or list of Tensors.")
            label = label.to(self.device)
            if train:
                self.optimizer.zero_grad()
                output = self.model(img)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    output = self.model(img)
                    loss = self.criterion(output, label)

            ctx = {"model": self.model, "max_spike_time": self.max_spike_time}
            correct, _ = self.acc_computer(output, label, ctx)
            num_acc += correct
            num_loss += loss.item()
            num_sample += label.size(0)

            batch_acc = correct * 1.0 / label.size(0)
            batch_loss = loss.item() / label.size(0)
            self.model.batch_reset()
            pbar.set_postfix({"ACC": f"{batch_acc:.4f}", "LOSS": f"{batch_loss:.4f}"})
        return num_sample, num_acc, num_loss

    def _plot_run(self, dataloader: torch.utils.data.DataLoader) -> None:
        """
        Run last epoch for plot some figs.
        Args:
            dataloader:  Dataloader for plotting.
        """
        num_classes = self.config["dataloader"]["num_classes"]
        cm = np.zeros((num_classes, num_classes))

        self.model.eval()
        for data, labels in dataloader:
            if isinstance(data, torch.Tensor):
                data = data.to(self.device)
            elif isinstance(data, list):
                data = [i.to(self.device) for i in data]
            else:
                raise TypeError(
                    f"Unsupported data type {type(data)} in dataloader, should be Tensor or list of Tensors.")
            labels = labels.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, labels)
            ctx = {"model": self.model, "max_spike_time": self.max_spike_time}
            acc, predictions = self.acc_computer(output, labels, ctx)
            self.model.batch_reset()
            # confusion matrix
            for pred, label in zip(predictions, labels):
                cm[label.item(), pred.item()] += 1
        self._plot_confusion_matrix(self.save_root, cm)
        self._plot_computation_graph(self.save_root, self.model, loss)

    def save_config(self) -> None:
        """
        Save all configuration.
        """
        save_path = os.path.join(self.save_root, "config.yaml")
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True)
        logger.info(f"Config saved in {save_path}")

    def save_model(self) -> None:
        """
        Save the current checkpoint.
        """
        path = os.path.join(self.save_root, "ckpt.pth")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        logger.info(f"Save checkpoint at {path}")
        torch.save(self.model.state_dict(), path)


class FLTimeStepTrainer:
    def __init__(self, epochs: int, config: dict, train_dataset_list: Dataset, test_dataset: Dataset,
                 debug: bool = False,save_root="./log"):
        """
        Args:
            config: See the description in BaseSimulator.
            train_dataset: See the description in BaseSimulator.
            test_dataset: See the description in BaseSimulator.
            debug: See the description in BaseSimulator.
        """
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        time_str = f"fl-SNN-{time_str}"
        self.save_root = os.path.join(save_root, time_str)
        self.epochs = epochs
        self.simulator_list = []
        for i in range(len(train_dataset_list)):
            self.simulator_list.append(TimeStepBasedSimulator(config, train_dataset_list[i], test_dataset, debug=debug))
            self.simulator_list[i].save_root = self.save_root
        self.dataset_length = [len(dataset) for dataset in train_dataset_list]
        self.dataset_size = np.array(self.dataset_length) / np.sum(self.dataset_length)
        self.result = {f"simulator_{i}": {"train_acc": [], "train_loss": [], "test_acc": [], "test_loss": []} for i in
                       range(len(self.simulator_list))}
        self.result[f"simulator_-1"] = {"train_acc": [], "train_loss": [], "test_acc": [], "test_loss": []}
        self.global_model = None
        self.num_delta_weight = defaultdict(list)

    def run_fl(self, merge: bool):
        if self.global_model is None:
            self.global_model = copy.deepcopy(self.simulator_list[0].model)
        for epoch in range(self.epochs):
            old_state_dict  = {k: v.clone().detach() for k, v in self.global_model.state_dict().items()}

            for i, sim in enumerate(self.simulator_list):
                self._epoch_run(i, epoch, sim, train=True)
                self._epoch_run(i, epoch, sim, train=False)
                new_state_dict = {k: v.clone().detach() for k, v in
                                                   self.simulator_list[i].model.state_dict().items()}

                changed, total = self.compute_num_weight_change(old_state_dict, new_state_dict)
                self.num_delta_weight[f"simulator_{i}"].append(changed)
            self.aggregate()
            if merge:
                self.update_edge_model()
                self._epoch_run(-1, epoch, self.simulator_list[0], train=False)
        self.plot_result()
        self.plot_weight_change()

    def plot_weight_change(self):
        """
        Plot the weight change of each simulator.
        """
        plt.figure(figsize=(10, 5))
        for i in range(len(self.simulator_list)):
            plt.plot(self.num_delta_weight[f"simulator_{i}"], label=f"Simulator {i}")
        total_change = np.sum(np.array(self.num_delta_weight.values()), axis=0)
        plt.plot(total_change, label="Total change")
        plt.xlabel("Epoch")
        plt.ylabel("Number of Changed Weights")
        plt.title("Weight Change During Training")
        plt.legend()
        save_path = os.path.join(self.save_root, "fig/weight_change.png")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        plt.savefig(save_path)
        plt.show()

        save_path = os.path.join(self.save_root, "data/weight_change.csv")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        df = pd.DataFrame(self.num_delta_weight)
        df.to_csv(save_path, index=False)

    def plot_result(self):
        """
        Plot the training result of all simulators.
        """
        for i, sim in enumerate(self.simulator_list):
            train_acc = self.result[f"simulator_{i}"]["train_acc"]
            train_loss = self.result[f"simulator_{i}"]["train_loss"]
            test_acc = self.result[f"simulator_{i}"]["test_acc"]
            test_loss = self.result[f"simulator_{i}"]["test_loss"]
            save_root = os.path.join(sim.save_root, f"simulator_{i}")
            sim._plot_acc(save_root, train_acc, train_loss, test_acc, test_loss)

        train_acc = self.result[f"simulator_-1"]["train_acc"]
        train_loss = self.result[f"simulator_-1"]["train_loss"]
        test_acc = self.result[f"simulator_-1"]["test_acc"]
        test_loss = self.result[f"simulator_-1"]["test_loss"]
        save_root = os.path.join(sim.save_root, f"simulator_-1")
        self.simulator_list[0]._plot_acc(save_root, train_acc, train_loss, test_acc, test_loss)

    def _epoch_run(self, id: int, epoch: int, simulator: TimeStepBasedSimulator, train: bool):
        if train:

            simulator.model.train()
            num_sample, num_acc, num_loss = simulator._epoch_run(simulator.train_loader, train=train)
        else:
            simulator.model.eval()
            num_sample, num_acc, num_loss = simulator._epoch_run(simulator.test_loader, train=train)
        epoch_acc = num_acc / num_sample
        epoch_loss = num_loss / num_sample
        if train:
            logger.info(
                f"Epoch {epoch + 1} Simulator {id}: train_acc: {epoch_acc:.4f},train_loss: {epoch_loss:.4f} ")
            self.result[f"simulator_{id}"]["train_acc"].append(epoch_acc)
            self.result[f"simulator_{id}"]["train_loss"].append(epoch_loss)
        else:
            logger.info(
                f"Epoch {epoch + 1} Simulator {id}: test_acc: {epoch_acc:.4f},test_loss: {epoch_loss:.4f} ")
            self.result[f"simulator_{id}"]["test_acc"].append(epoch_acc)
            self.result[f"simulator_{id}"]["test_loss"].append(epoch_loss)

    def aggregate(self):
        if self.global_model is None:
            self.global_model = copy.deepcopy(self.simulator_list[0].model)
        global_dict = self.global_model.state_dict()
        with torch.no_grad():
            for key in global_dict.keys():
                params = [sim.model.state_dict()[key] * self.dataset_size[i] for i, sim in
                          enumerate(self.simulator_list)]
                global_dict[key] = torch.stack(params, dim=0).sum(dim=0)

        self.global_model.load_state_dict(global_dict)

    def update_edge_model(self):
        for i, sim in enumerate(self.simulator_list):
            sim.model.load_state_dict(self.global_model.state_dict())

    @staticmethod
    def compute_num_weight_change(old_state_dict, new_state_dict, threshold=1e-6):
        changed = 0
        total = 0
        for key in old_state_dict:
            w1 = old_state_dict[key]
            w2 = new_state_dict[key]
            if not torch.is_floating_point(w1):
                continue
            diff = torch.abs(w1 - w2)
            changed += torch.sum(diff > threshold).item()
            total += w1.numel()
        return changed, total
