
from dataclasses import dataclass
from typing import List


@dataclass
class Axis:
    name: str
    reduction: bool = False

@dataclass
class Buffer:
    name: str
    axes: List[Axis]

@dataclass
class ComputeDescriptor:
    axes: List[Axis]
    input_buffers: List[Buffer]
    output_buffers: List[Buffer]