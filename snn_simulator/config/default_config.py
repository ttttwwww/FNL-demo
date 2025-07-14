"""
A default config for simulator,dataset and network
"""

#TODO delete this and change the way simulator save config


DEFAULT_CONFIG={
    "model":{
        "load_path":None,
        "type": None,
        "model_params":{
            # "input_size": None,
            # "hidden_sizes": None,
            # "output_size": None,
            # "init_weight_uppers": None,
            # "neuron_cls": None,
            # "neuron_params": {
            #
            #     "volt_reset": 0,
            #     "tau": 1e-7,
            #     "R1": 10e3,
            #     "Rh": 20e3,
            #     "Rl": 100,
            #     "C": 1e-9,
            #     "surrogate_function": "torch.heaviside",
            # },
            "surrogate_cls": None,
            "surrogate_params": {},
            "quantizer": None,

        },
    },
    "dataset":{
        "encoder_cls": None,
        "encoder_params": {},
    },
    "dataloader":{
        "num_classes": 10,
        "max_spike_time": 256,
        "batch_size": 64,
        "train_shuffle": True,
        "test_shuffle": False,
        "val_shuffle": False,
        "num_workers": 4,
        "train_size": 0.7,
        "test_size": 0.3,
    },
    "common":{
        "save_root": "./log",
        "epochs": 100,
        "device": "cuda",
        "optimizer": {
            "type": "torch.optim.Adam,",
            "lr": 0.01,
        },
        "scheduler": None,
        "criterion": {
            "type": "snn_simulator.loss.temporal_loss.S4NNLoss",
        },
        "acc_computer":"snn_simulator.utils.loss_utils.ttfs_acc_count"
    }
}