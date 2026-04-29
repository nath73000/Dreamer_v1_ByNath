import os
import csv
import yaml
import torch.nn as nn
import torch


class AttrDict(dict):
    def __init__(self, mapping):
        super().__init__()
        for key, value in mapping.items():
            self[key] = self._convert(value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @classmethod
    def _convert(cls, value):
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        return value


def find_file(filename):
    currentDir = os.getcwd()
    for root, dirs, files in os.walk(currentDir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"File '{filename}' not found in subdirectories of {currentDir}")


def load_config(config_name):
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"

    config_path = config_name if os.path.isfile(config_name) else find_file(os.path.basename(config_name))
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return AttrDict(config)


def get_env_properties(env):
    observation_spec     = env.observation_spec()
    action_spec          = env.action_spec()

    observation_shape    = observation_spec["pixels"].shape
    action_size          = int(torch.tensor(action_spec.shape).prod().item())
    action_min           = action_spec.minimum
    action_max           = action_spec.maximum

    return observation_shape, action_size, action_min, action_max


def creat_sequential_model_1D(input_size, hidden_sizes, output_size, activation_function, finishWithActivation=False):
    if isinstance(activation_function, str):
        activation_function = getattr(nn, activation_function)
    layers = []
    current_input_size = input_size

    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(current_input_size, hidden_size))
        layers.append(activation_function())
        current_input_size = hidden_size

    layers.append(nn.Linear(current_input_size, output_size))
    if finishWithActivation:
        layers.append(activation_function())

    return nn.Sequential(*layers)


def computeLambdaValues(rewards, values, continues, lambda_=0.95):
    returns = torch.zeros_like(rewards)
    bootstrap = values[:, -1]
    for i in reversed(range(rewards.shape[-1])):
        returns[:, i] = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i] + lambda_ * bootstrap)
        bootstrap = returns[:, i]
    return returns


class Moments(nn.Module):
    def __init__( self, device, decay = 0.99, min_=1, percentileLow = 0.05, percentileHigh = 0.95):
        super().__init__()
        self._decay = decay
        self._percentileLow = percentileLow
        self._percentileHigh = percentileHigh
        self.register_buffer("min_value", torch.tensor(min_, dtype=torch.float32, device=device))
        self.register_buffer("low", torch.zeros((), dtype=torch.float32, device=device))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.detach()
        low = torch.quantile(x, self._percentileLow)
        high = torch.quantile(x, self._percentileHigh)
        self.low = self._decay*self.low + (1 - self._decay)*low
        self.high = self._decay*self.high + (1 - self._decay)*high
        inverseScale = torch.max(self.min_value, self.high - self.low)
        return self.low.detach(), inverseScale.detach()
    

def saveLossesToCSV(filename, metrics):
    csv_path = filename if filename.endswith(".csv") else filename + ".csv"
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fileAlreadyExists = os.path.isfile(csv_path)
    fieldnames = list(metrics.keys())
    existing_rows = []
    rewrite_file = False
    if fileAlreadyExists:
        with open(csv_path, newline='') as file:
            reader = csv.DictReader(file)
            existing_fieldnames = reader.fieldnames or []
            if existing_fieldnames != fieldnames:
                fieldnames = list(dict.fromkeys(existing_fieldnames + fieldnames))
                existing_rows = list(reader)
                rewrite_file = True

    with open(csv_path, mode='w' if rewrite_file else 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not fileAlreadyExists or rewrite_file:
            writer.writeheader()
        if rewrite_file:
            writer.writerows(existing_rows)
        writer.writerow(metrics)


def plotMetrics(filename, title="", savePath="metricsPlot", window=10):
    import pandas as pd
    import plotly.graph_objects as pgo

    if not filename.endswith(".csv"):
        filename += ".csv"
    if not savePath.endswith(".html"):
        savePath += ".html"

    directory = os.path.dirname(savePath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    data = pd.read_csv(filename)
    fig = pgo.Figure()
    x_column = "gradient_steps" if "gradient_steps" in data.columns else "gradientSteps"

    colors = [
        "gold", "gray", "beige", "blueviolet", "cadetblue",
        "chartreuse", "coral", "cornflowerblue", "crimson", "darkorange",
        "deeppink", "dodgerblue", "forestgreen", "aquamarine", "lightseagreen",
        "lightskyblue", "mediumorchid", "mediumspringgreen", "orangered", "violet"]
    num_colors = len(colors)

    for idx, column in enumerate(data.columns):
        if column in ["env_steps", "gradient_steps", "envSteps", "gradientSteps"]:
            continue
        
        fig.add_trace(pgo.Scatter(
            x=data[x_column], y=data[column], mode='lines',
            name=f"{column} (original)",
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5, visible='legendonly'))
        
        smoothed_data = data[column].rolling(window=window, min_periods=1).mean()
        fig.add_trace(pgo.Scatter(
            x=data[x_column], y=smoothed_data, mode='lines',
            name=f"{column} (smoothed)",
            line=dict(color=colors[idx % num_colors], width=2)))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=30),
            yanchor='top'
        ),
        xaxis=dict(
            title="Gradient Steps",
            showgrid=True,
            zeroline=False,
            position=0
        ),
        yaxis_title="Value",
        template="plotly_dark",
        height=1080,
        width=1920,
        margin=dict(t=60, l=40, r=40, b=40),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="White",
            borderwidth=2,
            font=dict(size=12)
        )
    )

    fig.write_html(savePath)
