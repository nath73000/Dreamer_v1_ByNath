import os
import yaml
import attridict
import torch.nn as nn

# ------------------------------------------------------------- #
def find_file(filename):
    currentDir = os.getcwd()
    for root, dirs, files in os.walk(currentDir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"File '{filename}' not found in subdirectories of {currentDir}")


def load_config(config_name):
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    config_path = find_file(config_name)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return attridict(config)


def get_env_properties(env):
    observation_spec     = env.observation_spec()
    action_spec          = env.action_spec()

    observation_shape    = observation_spec["pixels"].shape
    action_size          = action_spec.shape
    action_min           = action_spec.minimum
    action_max           = action_spec.maximum

    return observation_shape, action_size, action_min, action_max


def creat_sequential_model_1D(input_size, hidden_sizes, output_size, activation_function, finishWithActivation=False):
    activation_function = getattr(nn, activation_function)
    layers = []
    current_input_size = input_size

    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(current_input_size, hidden_size))
        layers.append(activation_function)
        current_input_size = hidden_size

    layers.append(nn.Linear(current_input_size, output_size))
    if finishWithActivation:
        layers.append(activation_function)

    return nn.Sequential(layers)