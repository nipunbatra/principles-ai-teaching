from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LinearLayerSpec:
    in_features: int
    out_features: int
    activation: str = "Identity"
    name: str = ""


@dataclass
class TracedLinearLayer:
    linear: nn.Linear
    activation: str = "Identity"
    name: str = ""


@dataclass
class ForwardLayerValues:
    input_activation: torch.Tensor
    z: torch.Tensor
    a: torch.Tensor


SUPPORTED_ACTIVATION_MODULES: tuple[type[nn.Module], ...] = (
    nn.ReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.LeakyReLU,
    nn.GELU,
    nn.ELU,
    nn.Softmax,
    nn.LogSoftmax,
    nn.Identity,
)

IGNORED_MODULES: tuple[type[nn.Module], ...] = (
    nn.Flatten,
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.Identity,
)

SUPPORTED_FUNCTIONAL_ACTIVATIONS: dict[Callable, str] = {
    torch.relu: "ReLU",
    torch.sigmoid: "Sigmoid",
    torch.softmax: "Softmax",
    torch.tanh: "Tanh",
    F.relu: "ReLU",
    F.gelu: "GELU",
    F.softmax: "Softmax",
}

INPUT_COLOR = "#dff2d8"
HIDDEN_COLOR = "#e3e0ff"
OUTPUT_COLOR = "#f8dddd"
HEADER_COLOR = "#1d4ed8"
EDGE_COLOR = "#111827"
HIGHLIGHT_COLORS = ("#ef4444", "#10b981", "#3b82f6", "#f59e0b", "#8b5cf6", "#ec4899")


def _activation_name_from_module(module: nn.Module) -> str | None:
    if isinstance(module, nn.Identity):
        return "Identity"
    for activation_type in SUPPORTED_ACTIVATION_MODULES:
        if isinstance(module, activation_type):
            return activation_type.__name__
    return None


def _activation_name_from_function(target: Callable) -> str | None:
    return SUPPORTED_FUNCTIONAL_ACTIVATIONS.get(target)


def _trace_fully_connected_layers(model: nn.Module) -> list[TracedLinearLayer]:
    traced = fx.symbolic_trace(model)
    modules = dict(traced.named_modules())
    linear_layers: list[TracedLinearLayer] = []

    for node in traced.graph.nodes:
        if node.op == "call_module":
            module = modules[node.target]

            if isinstance(module, nn.Linear):
                linear_layers.append(
                    TracedLinearLayer(
                        linear=module,
                        activation="Identity",
                        name=node.target,
                    )
                )
                continue

            activation_name = _activation_name_from_module(module)
            if activation_name is not None:
                if linear_layers:
                    linear_layers[-1].activation = activation_name
                continue

            if isinstance(module, IGNORED_MODULES):
                continue

            raise ValueError(
                f"Unsupported module in fully connected visualizer: {node.target} ({type(module).__name__})"
            )

        if node.op == "call_function":
            activation_name = _activation_name_from_function(node.target)
            if activation_name is not None:
                if linear_layers:
                    linear_layers[-1].activation = activation_name
                continue
            raise ValueError(f"Unsupported function in fully connected visualizer: {node.target}")

        if node.op == "call_method":
            if node.target in {"view", "reshape", "flatten", "squeeze", "unsqueeze", "permute", "transpose"}:
                continue
            raise ValueError(f"Unsupported tensor method in fully connected visualizer: {node.target}")

        if node.op in {"placeholder", "output", "get_attr"}:
            continue

        raise ValueError(f"Unsupported FX node type: {node.op}")

    if not linear_layers:
        raise ValueError("No nn.Linear layers found. This visualizer only supports fully connected networks.")

    return linear_layers


def extract_fully_connected_architecture(model: nn.Module) -> list[LinearLayerSpec]:
    traced_layers = _trace_fully_connected_layers(model)
    return [
        LinearLayerSpec(
            in_features=layer.linear.in_features,
            out_features=layer.linear.out_features,
            activation=layer.activation,
            name=layer.name,
        )
        for layer in traced_layers
    ]


def _display_indices(size: int, max_drawn: int) -> list[int | None]:
    if size <= max_drawn:
        return list(range(size))

    front = max_drawn // 2
    back = max_drawn - front - 1
    indices: list[int | None] = list(range(front))
    indices.append(None)
    indices.extend(range(size - back, size))
    return indices


def _layer_role(layer_index: int, total_layers: int) -> str:
    if layer_index == 0:
        return "input"
    if layer_index == total_layers - 1:
        return "output"
    return "hidden"


def _layer_box_label(layer_index: int, total_layers: int) -> str:
    role = _layer_role(layer_index, total_layers)
    if role == "input":
        return "Input layer"
    if role == "output":
        return "Output layer"
    return f"Hidden layer {layer_index}"


def _layer_color(layer_index: int, total_layers: int) -> str:
    role = _layer_role(layer_index, total_layers)
    if role == "input":
        return INPUT_COLOR
    if role == "output":
        return OUTPUT_COLOR
    return HIDDEN_COLOR


def _format_input_label(index: int, input_dim: int, input_labels: list[str] | None) -> str:
    if input_labels and index < len(input_labels):
        return input_labels[index]
    if input_dim == 1:
        return r"$x$"
    return rf"$x_{{{index + 1}}}$"


def _format_output_label(index: int, output_dim: int, output_labels: list[str] | None) -> str:
    if output_labels and index < len(output_labels):
        return output_labels[index]
    if output_dim == 1:
        return r"$\hat{y}$"
    return rf"$\hat{{y}}_{{{index + 1}}}$"


def _draw_split_node(
    ax: plt.Axes,
    x: float,
    y: float,
    radius: float,
    facecolor: str,
    left_label: str,
    right_label: str,
    fontsize: float,
    edgecolor: str = "black",
    linewidth: float = 1.5,
) -> None:
    ax.add_patch(Circle((x, y), radius, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=5))
    ax.plot([x, x], [y - radius * 0.88, y + radius * 0.88], color="black", linewidth=1.1, zorder=6)
    ax.text(x - radius * 0.34, y, left_label, ha="center", va="center", fontsize=fontsize, zorder=7)
    ax.text(x + radius * 0.34, y, right_label, ha="center", va="center", fontsize=fontsize, zorder=7)


def _node_accent_color(layer_index: int, neuron_index: int) -> str:
    return HIGHLIGHT_COLORS[(layer_index * 7 + neuron_index) % len(HIGHLIGHT_COLORS)]


def _tint(color: str, mix: float = 0.82) -> tuple[float, float, float]:
    base = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    return tuple(base * (1 - mix) + white * mix)


def _draw_badge(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    fontsize: float = 8.0,
    ha: str = "center",
    va: str = "center",
    alpha: float = 0.98,
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.18,rounding_size=0.15",
            "facecolor": _tint(color, 0.82),
            "edgecolor": color,
            "linewidth": 0.9,
            "alpha": alpha,
        },
        zorder=9,
    )


def _apply_activation(name: str, values: torch.Tensor) -> torch.Tensor:
    if name == "Identity":
        return values
    if name == "ReLU":
        return torch.relu(values)
    if name == "Sigmoid":
        return torch.sigmoid(values)
    if name == "Tanh":
        return torch.tanh(values)
    if name == "Softmax":
        return torch.softmax(values, dim=-1)
    if name == "LogSoftmax":
        return torch.log_softmax(values, dim=-1)
    if name == "GELU":
        return F.gelu(values)
    if name == "LeakyReLU":
        return F.leaky_relu(values)
    if name == "ELU":
        return F.elu(values)
    raise ValueError(f"Unsupported activation for forward-value display: {name}")


def _compute_forward_layer_values(
    traced_layers: list[TracedLinearLayer],
    input_vector: torch.Tensor | np.ndarray | list[float] | tuple[float, ...],
) -> tuple[torch.Tensor, list[ForwardLayerValues]]:
    input_activation = torch.as_tensor(input_vector, dtype=torch.float32).reshape(-1)
    expected_dim = traced_layers[0].linear.in_features
    if input_activation.numel() != expected_dim:
        raise ValueError(f"Expected input_vector with {expected_dim} values, got {input_activation.numel()}.")

    current_activation = input_activation
    layer_values: list[ForwardLayerValues] = []

    for traced_layer in traced_layers:
        z = traced_layer.linear(current_activation)
        a = _apply_activation(traced_layer.activation, z)
        layer_values.append(
            ForwardLayerValues(
                input_activation=current_activation.detach().clone(),
                z=z.detach().clone(),
                a=a.detach().clone(),
            )
        )
        current_activation = a

    return input_activation.detach().clone(), layer_values


def _format_scalar(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _format_array(values: torch.Tensor | np.ndarray, decimals: int) -> str:
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    return np.array2string(array, precision=decimals, suppress_small=True)


def _build_parameter_panel_text(
    traced_layers: list[TracedLinearLayer],
    input_activation: torch.Tensor | None,
    forward_values: list[ForwardLayerValues] | None,
    decimals: int,
) -> str:
    blocks: list[str] = []

    if input_activation is not None:
        blocks.append(f"a[0] = {_format_array(input_activation, decimals)}")

    for layer_number, traced_layer in enumerate(traced_layers, start=1):
        linear = traced_layer.linear
        lines = [
            f"Layer {layer_number}",
            f"z[{layer_number}] = W[{layer_number}] a[{layer_number - 1}] + b[{layer_number}]",
            f"a[{layer_number}] = {traced_layer.activation}(z[{layer_number}])",
            f"W[{layer_number}] =",
            _format_array(linear.weight, decimals),
            f"b[{layer_number}] = {_format_array(linear.bias, decimals)}",
        ]
        if forward_values is not None:
            values = forward_values[layer_number - 1]
            lines.append(f"z[{layer_number}] = {_format_array(values.z, decimals)}")
            lines.append(f"a[{layer_number}] = {_format_array(values.a, decimals)}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def visualize_fully_connected_model(
    model: nn.Module,
    *,
    input_labels: list[str] | None = None,
    output_labels: list[str] | None = None,
    title: str = "PyTorch Fully Connected Network",
    max_neurons_per_layer: int = 5,
    input_vector: torch.Tensor | np.ndarray | list[float] | tuple[float, ...] | None = None,
    show_edge_weights: bool | None = None,
    show_biases: bool | None = None,
    show_values: bool | None = None,
    show_matrix_details: bool | None = None,
    decimals: int = 2,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize a fully connected PyTorch network in a lecture-style layer diagram."""
    traced_layers = _trace_fully_connected_layers(model)
    layer_sizes = [traced_layers[0].linear.in_features] + [layer.linear.out_features for layer in traced_layers]

    display_rows = [_display_indices(size, max_neurons_per_layer) for size in layer_sizes]
    max_rows = max(len(rows) for rows in display_rows)
    total_layers = len(layer_sizes)
    all_nodes_drawn = all(len(displayed) == size for displayed, size in zip(display_rows, layer_sizes))
    edge_count = sum(layer_sizes[i] * layer_sizes[i + 1] for i in range(total_layers - 1))
    parameter_count = sum(layer.linear.weight.numel() + layer.linear.bias.numel() for layer in traced_layers)
    small_network = all_nodes_drawn and max(layer_sizes) <= 8 and edge_count <= 20

    if show_edge_weights is None:
        show_edge_weights = small_network
    if show_biases is None:
        show_biases = small_network
    if show_values is None:
        show_values = small_network and input_vector is not None
    if show_matrix_details is None:
        show_matrix_details = small_network and parameter_count <= 40

    input_activation: torch.Tensor | None = None
    forward_values: list[ForwardLayerValues] | None = None
    if input_vector is not None:
        input_activation, forward_values = _compute_forward_layer_values(traced_layers, input_vector)

    if show_values and forward_values is None:
        raise ValueError("show_values=True requires input_vector.")

    detailed_small_mode = small_network and (show_edge_weights or show_biases or show_values)
    panel_width = 5.3 if show_matrix_details else 0.0
    row_spacing = 1.75 if detailed_small_mode and all_nodes_drawn else 1.15
    layer_gap = 3.6 if detailed_small_mode else 3.0
    box_width = 2.45 if detailed_small_mode else 2.2

    if figsize is None:
        figsize = ((layer_gap + 0.2) * total_layers + 1.1 + panel_width, row_spacing * max_rows + 2.9)

    fig, ax = plt.subplots(figsize=figsize)

    radius = 0.34
    box_bottom = 0.6
    box_height = max_rows * row_spacing + 1.2
    box_top = box_bottom + box_height
    x_positions = [1.6 + i * layer_gap for i in range(total_layers)]

    neuron_positions: dict[tuple[int, int], tuple[float, float]] = {}

    for layer_index, x_pos in enumerate(x_positions):
        ax.add_patch(
            Rectangle(
                (x_pos - box_width / 2, box_bottom),
                box_width,
                box_height,
                facecolor="none",
                edgecolor="black",
                linewidth=1.8,
                zorder=0,
            )
        )

        ax.text(
            x_pos,
            box_top - 0.25,
            f"LAYER {layer_index}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color=HEADER_COLOR,
        )

        if layer_index > 0:
            ax.text(
                x_pos,
                box_top + 0.25,
                traced_layers[layer_index - 1].activation,
                ha="center",
                va="bottom",
                fontsize=11,
                color="#374151",
                fontweight="bold",
            )

        ax.text(
            x_pos,
            box_bottom + 0.22,
            _layer_box_label(layer_index, total_layers),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

        displayed = display_rows[layer_index]
        start_y = box_top - 1.15
        layer_color = _layer_color(layer_index, total_layers)

        for row_index, neuron_index in enumerate(displayed):
            y_pos = start_y - row_index * row_spacing

            if neuron_index is None:
                ax.text(x_pos, y_pos, "...", rotation=90, ha="center", va="center", fontsize=18, color="#4b5563")
                continue

            neuron_positions[(layer_index, neuron_index)] = (x_pos, y_pos)
            role = _layer_role(layer_index, total_layers)
            accent_color = _node_accent_color(layer_index, neuron_index) if role != "input" else "#64748b"

            if role == "input":
                ax.add_patch(
                    Circle((x_pos, y_pos), radius, facecolor=layer_color, edgecolor="black", linewidth=1.5, zorder=5)
                )
                ax.text(
                    x_pos,
                    y_pos,
                    _format_input_label(neuron_index, layer_sizes[0], input_labels),
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    zorder=7,
                )
                if show_values and input_activation is not None:
                    _draw_badge(
                        ax,
                        x_pos,
                        y_pos - radius - 0.12,
                        f"a={_format_scalar(input_activation[neuron_index].item(), decimals)}",
                        color="#94a3b8",
                        fontsize=8.2,
                        va="top",
                    )
                continue

            if role == "output":
                if layer_sizes[-1] == 1:
                    _draw_split_node(
                        ax,
                        x_pos,
                        y_pos,
                        radius,
                        layer_color,
                        left_label=rf"$z^{{[{layer_index}]}}$",
                        right_label=rf"$a^{{[{layer_index}]}}$",
                        fontsize=8,
                        edgecolor=accent_color,
                        linewidth=2.0 if detailed_small_mode else 1.5,
                    )
                    ax.annotate(
                        "",
                        xy=(x_pos + 0.95, y_pos),
                        xytext=(x_pos + radius, y_pos),
                        arrowprops=dict(arrowstyle="->", linewidth=2, color="black"),
                    )
                    ax.text(
                        x_pos + 1.12,
                        y_pos,
                        _format_output_label(neuron_index, layer_sizes[-1], output_labels),
                        ha="left",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                    )
                else:
                    _draw_split_node(
                        ax,
                        x_pos,
                        y_pos,
                        radius,
                        layer_color,
                        left_label=rf"$z^{{[{layer_index}]}}_{{{neuron_index + 1}}}$",
                        right_label=rf"$a^{{[{layer_index}]}}_{{{neuron_index + 1}}}$",
                        fontsize=7,
                        edgecolor=accent_color,
                        linewidth=2.0 if detailed_small_mode else 1.5,
                    )
                if show_values and forward_values is not None:
                    node_values = forward_values[layer_index - 1]
                    _draw_badge(
                        ax,
                        x_pos,
                        y_pos + radius + 0.08,
                        f"z={_format_scalar(node_values.z[neuron_index].item(), decimals)}",
                        color=accent_color,
                        fontsize=8.2,
                        va="bottom",
                    )
                    _draw_badge(
                        ax,
                        x_pos,
                        y_pos - radius - 0.12,
                        f"a={_format_scalar(node_values.a[neuron_index].item(), decimals)}",
                        color=accent_color,
                        fontsize=8.2,
                        va="top",
                    )
                if show_biases:
                    bias_value = traced_layers[layer_index - 1].linear.bias[neuron_index].item()
                    _draw_badge(
                        ax,
                        x_pos + radius + 0.26,
                        y_pos,
                        f"b={_format_scalar(bias_value, decimals)}",
                        color=accent_color,
                        fontsize=8.0,
                        ha="left",
                    )
                continue

            _draw_split_node(
                ax,
                x_pos,
                y_pos,
                radius,
                layer_color,
                left_label=rf"$z^{{[{layer_index}]}}_{{{neuron_index + 1}}}$",
                right_label=rf"$a^{{[{layer_index}]}}_{{{neuron_index + 1}}}$",
                fontsize=7,
                edgecolor=accent_color,
                linewidth=2.0 if detailed_small_mode else 1.5,
            )
            if show_values and forward_values is not None:
                node_values = forward_values[layer_index - 1]
                _draw_badge(
                    ax,
                    x_pos,
                    y_pos + radius + 0.08,
                    f"z={_format_scalar(node_values.z[neuron_index].item(), decimals)}",
                    color=accent_color,
                    fontsize=8.2,
                    va="bottom",
                )
                _draw_badge(
                    ax,
                    x_pos,
                    y_pos - radius - 0.12,
                    f"a={_format_scalar(node_values.a[neuron_index].item(), decimals)}",
                    color=accent_color,
                    fontsize=8.2,
                    va="top",
                )
            if show_biases:
                bias_value = traced_layers[layer_index - 1].linear.bias[neuron_index].item()
                _draw_badge(
                    ax,
                    x_pos + radius + 0.26,
                    y_pos,
                    f"b={_format_scalar(bias_value, decimals)}",
                    color=accent_color,
                    fontsize=8.0,
                    ha="left",
                )

    for layer_index in range(total_layers - 1):
        left_indices = [index for index in display_rows[layer_index] if index is not None]
        right_indices = [index for index in display_rows[layer_index + 1] if index is not None]
        for left_index in left_indices:
            for right_index in right_indices:
                x1, y1 = neuron_positions[(layer_index, left_index)]
                x2, y2 = neuron_positions[(layer_index + 1, right_index)]
                accent_color = _node_accent_color(layer_index + 1, right_index)
                edge_color = accent_color if detailed_small_mode else EDGE_COLOR
                ax.plot(
                    [x1 + radius, x2 - radius],
                    [y1, y2],
                    color=edge_color,
                    alpha=0.78 if not detailed_small_mode else 0.72,
                    linewidth=1.5 if not detailed_small_mode else 1.8,
                    zorder=1,
                )
                if show_edge_weights:
                    weight_value = traced_layers[layer_index].linear.weight[right_index, left_index].item()
                    fraction = 0.68
                    mid_x = x1 + (x2 - x1) * fraction
                    mid_y = y1 + (y2 - y1) * fraction
                    dx = x2 - x1
                    dy = y2 - y1
                    length = float(np.hypot(dx, dy))
                    if length == 0:
                        offset_x, offset_y = 0.0, 0.0
                    else:
                        spread = left_indices.index(left_index) - (len(left_indices) - 1) / 2
                        offset_scale = 0.10 + 0.06 * spread
                        offset_x = -dy / length * offset_scale
                        offset_y = dx / length * offset_scale
                    _draw_badge(
                        ax,
                        mid_x + offset_x,
                        mid_y + offset_y,
                        _format_scalar(weight_value, decimals),
                        color=accent_color,
                        fontsize=7.2,
                    )

    if show_matrix_details:
        panel_text = _build_parameter_panel_text(traced_layers, input_activation, forward_values, decimals)
        panel_x = x_positions[-1] + box_width / 2 + 0.7
        ax.text(
            panel_x,
            box_top,
            panel_text,
            ha="left",
            va="top",
            fontsize=8.4,
            family="monospace",
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.45,rounding_size=0.2",
                "facecolor": "#f8fafc",
                "edgecolor": "#94a3b8",
                "linewidth": 1.2,
            },
            zorder=10,
        )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=18)
    x_max = x_positions[-1] + box_width / 2 + 1.8 + panel_width
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, box_top + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def save_figure(fig: plt.Figure, png_path: Path, svg_path: Path | None = None) -> None:
    if svg_path is None:
        svg_path = png_path.with_suffix(".svg")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_demo_model() -> nn.Module:
    class DemoFCNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = nn.Linear(4, 3)
            self.act1 = nn.ReLU()
            self.layer2 = nn.Linear(3, 2)
            self.act2 = nn.Tanh()
            self.output = nn.Linear(2, 1)
            self.output_activation = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.act1(self.layer1(x))
            x = self.act2(self.layer2(x))
            return self.output_activation(self.output(x))

    return DemoFCNet()


def main() -> None:
    model = build_demo_model()
    input_vector = torch.tensor([0.6, -0.4, 0.2, 0.9], dtype=torch.float32)
    fig, _ = visualize_fully_connected_model(
        model,
        input_labels=[r"$x_1$", r"$x_2$", r"$x_3$", r"$x_d$"],
        title="Fully Connected PyTorch Network",
        max_neurons_per_layer=5,
        input_vector=input_vector,
    )

    base_dir = Path(__file__).resolve().parent / "diagrams"
    save_figure(
        fig,
        png_path=base_dir / "png" / "pytorch_fc_network_example.png",
        svg_path=base_dir / "svg" / "pytorch_fc_network_example.svg",
    )

    print("Saved diagrams/png/pytorch_fc_network_example.png")
    print("Saved diagrams/svg/pytorch_fc_network_example.svg")


if __name__ == "__main__":
    main()
