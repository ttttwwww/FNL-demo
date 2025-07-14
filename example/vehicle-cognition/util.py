import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt


def aggregate(model_list):
    """
    Aggregate the parameters of the models in model_list.
    """
    global_params = model_list[0].state_dict()
    for key in global_params.keys():
        global_params[key] = sum(model.state_dict()[key] for model in model_list) / len(model_list)
    return global_params

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

def plot_weight_change(weight_change_dict):

    for model_name, changes in weight_change_dict.items():
        plt.plot(changes, label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Number of Weight Changes')
    plt.title('Weight Changes Over Epochs')
    plt.legend()
    plt.show()


def plot_result(save_root, train_acc, train_loss, test_acc, test_loss) -> None:
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

def plot_weight_change(save_root,weight_change_dict):
    """
    Plot the weight change of each simulator.
    """
    plt.figure(figsize=(10, 5))
    for key,value in weight_change_dict.items():
        plt.plot(value, label=key)
    total_change = np.sum(np.array(weight_change_dict.values()), axis=0)
    plt.plot(total_change, label="Total change")
    plt.xlabel("Epoch")
    plt.ylabel("Number of Changed Weights")
    plt.title("Weight Change During Training")
    plt.legend()
    save_path = os.path.join(save_root, "fig/weight_change.png")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.show()

    save_path = os.path.join(save_root, "data/weight_change.csv")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    df = pd.DataFrame(weight_change_dict)
    df.to_csv(save_path, index=False)
