from snn_simulator.models.net import MnistTimeStepNet
from snn_simulator.utils.logger import setup_logger
setup_logger("snn_simulator","./test.log")

net_params = {
    "input_size": 784,
    "hidden_sizes": [400],
    "output_size": 10,
    "init_weight_uppers": [0.5,0.1],
    "neuron_cls": "snn_simulator.neuron.lif.RCLIF",
    "neuron_params" :{
        "volt_threshold": 2,
        "volt_reset": 0,
        "tau": 1e-7,
        "R1": 10e3,
        "Rh": 20e3,
        "Rl": 100,
        "C": 1e-9,
        "surrogate_function": "torch.heaviside",
    },
    "surrogate_cls": "snn_simulator.surrogate.layer_surrogate.S4NNSurrogate",
    "surrogate_params" :{
        "max_spike_time": 256,
    },
    "max_spike_time": 256,
    "quantizer": None ,}
model = MnistTimeStepNet(**net_params)
print(model)