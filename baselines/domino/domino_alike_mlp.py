import torch
import torch.distributed as dist
import torch.nn.functional as F
from dataclasses import dataclass
from torch.nn.parameter import Parameter
from typing import Callable, Optional, Tuple

from deepspeed.accelerator import get_accelerator


@dataclass
class DominoAlikeMLPConfig:
    hidden_size: int
    ffn_hidden_size: int
    params_dtype: torch.dtype
    init_method: Callable[[torch.Tensor], torch.Tensor]
    output_layer_init_method: Callable[[torch.Tensor], torch.Tensor]
    add_bias_linear: bool
    perform_initialization: bool = True


class DominoAsyncColumnParallelLinear(torch.nn.Module):
    """
    Column-parallel linear without forward-time communication.
    Backward will all-reduce grad_input following DeepSpeed Domino.
    """

    def __init__(self,
                 input_size: int,
                 output_size_per_partition: int,
                 config: DominoAlikeMLPConfig,
                 bias: bool = True):
        super().__init__()
        device = get_accelerator().current_device_name()
        self.weight = Parameter(torch.empty(output_size_per_partition, input_size, device=device, dtype=config.params_dtype))
        if config.perform_initialization:
            config.init_method(self.weight)

        if bias and config.add_bias_linear:
            self.bias = Parameter(torch.empty(output_size_per_partition, device=device, dtype=config.params_dtype))
            if config.perform_initialization:
                with torch.no_grad():
                    self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def forward(self, input_: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output = torch.matmul(input_, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output, self.bias


class RowParallelLinearNoComm(torch.nn.Module):

    def __init__(self,
                 input_size_per_partition: int,
                 output_size: int,
                 config: DominoAlikeMLPConfig,
                 bias: bool = True):
        super().__init__()
        device = get_accelerator().current_device_name()
        self.weight = Parameter(torch.empty(output_size, input_size_per_partition, device=device, dtype=config.params_dtype))
        if config.perform_initialization:
            config.output_layer_init_method(self.weight)

        if bias and config.add_bias_linear:
            self.bias = Parameter(torch.empty(output_size, device=device, dtype=config.params_dtype))
            if config.perform_initialization:
                with torch.no_grad():
                    self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def forward(self, input_: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output = F.linear(input_, self.weight, self.bias)
        return output, self.bias


class DominoAlikeMLP(torch.nn.Module):
    """
    Domino-style MLP block for two micro-batches, adjusted to only overlap the
    second GEMM with all-reduce (no fused GEMM+GELU+GEMM kernel available).
    """

    def __init__(self, config: DominoAlikeMLPConfig, tp_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.tp_group = tp_group
        tp_world_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1

        if config.ffn_hidden_size % tp_world_size != 0 or config.hidden_size % tp_world_size != 0:
            raise ValueError("hidden_size and ffn_hidden_size must be divisible by tensor-parallel world size.")

        output_size_per_partition = config.ffn_hidden_size // tp_world_size
        input_size_per_partition = config.ffn_hidden_size // tp_world_size

        self.linear_fc1 = DominoAsyncColumnParallelLinear(config.hidden_size,
                                                          output_size_per_partition,
                                                          config=config,
                                                          bias=config.add_bias_linear)
        self.mlp_activation_func = F.gelu
        self.linear_fc2 = RowParallelLinearNoComm(input_size_per_partition,
                                                  config.hidden_size,
                                                  config=config,
                                                  bias=config.add_bias_linear)

    def forward(self, hidden_states: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        hidden_states0, hidden_states1 = hidden_states

        # Stage 1: GEMM+GELU for micro-batch 0.
        output0, _ = self.linear_fc1(hidden_states0)
        output0 = self.mlp_activation_func(output0)

        # Stage 2: Second GEMM for micro-batch 0 + async all-reduce.
        hidden_states0, last_mlp_bias = self.linear_fc2(output0)
        handle0 = dist.all_reduce(hidden_states0, group=self.tp_group, async_op=True) if dist.is_initialized() else None

        # Stage 3: GEMM+GELU for micro-batch 1.
        output1, _ = self.linear_fc1(hidden_states1)
        output1 = self.mlp_activation_func(output1)

        # Stage 4: Second GEMM for micro-batch 1 while all-reduce 0 is in flight.
        hidden_states1, _ = self.linear_fc2(output1)
        handle1 = dist.all_reduce(hidden_states1, group=self.tp_group, async_op=True) if dist.is_initialized() else None

        if handle0 is not None:
            handle0.wait()
        if handle1 is not None:
            handle1.wait()

        return hidden_states0, hidden_states1, last_mlp_bias
