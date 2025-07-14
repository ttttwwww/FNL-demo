from snn_simulator.utils.logger import setup_logger
from snn_simulator.neuron.lif import RCLIF

setup_logger("","./neuron_test.log")

neuron_params = {
    "volt_threshold": 2,
    "volt_reset": 0,
    "tau": 1e-7,
    "R1": 10e3,
    "Rh": 20e3,
    "Rl": 100,
    "C": 1e-9,
    "surrogate_function": "torch.heaviside",
}



neuron = RCLIF(**neuron_params)
print(neuron)