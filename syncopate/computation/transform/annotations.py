from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class AxisCount:
    axis: str
    var: str
    kind: str  # "single" or "range"
    expr: str
    block: str


@dataclass
class DispatchBlock:
    start: int
    end: int
    indent: str


@dataclass
class ParsedAnnotations:
    tile_id_var: str
    tile_id_expr: str
    axis_counts: List[AxisCount]
    pid_map: Dict[str, str]
    dispatch: DispatchBlock
    persistent: bool
    persistent_init: Optional[DispatchBlock]
    producer_epilogue: Optional[DispatchBlock]


ASSIGNMENT_RE = re.compile(r"^\s*(?P<lhs>[^=]+?)\s*=\s*(?P<rhs>.+?)\s*$")


class AnnotationError(RuntimeError):
    """Raised when annotations are malformed or inconsistent."""


def _strip_comment(line: str) -> str:
    head, _, _ = line.partition("#")
    return head.rstrip()


def _parse_assignment(text: str) -> tuple[str, str]:
    match = ASSIGNMENT_RE.match(text)
    if not match:
        raise AnnotationError(f"Expected assignment, got: {text!r}")
    return match.group("lhs").strip(), match.group("rhs").strip()


def _parse_axis_count(comment: str) -> tuple[str, str, str]:
    payload = comment.split("@sy.axis_count", 1)[1].strip()
    block_expr = "1"
    block_match = re.search(r"\bblock\s*=\s*(.+)", payload)
    if block_match:
        block_expr = block_match.group(1).strip()
        payload = payload[: block_match.start()].rstrip()
        if not block_expr:
            raise AnnotationError("Empty block expression in @sy.axis_count annotation")
    payload = payload.strip()
    axis, sep, suffix = payload.partition("=")
    axis = axis.strip()
    kind = suffix.strip().lower() if sep else "single"
    if kind not in {"single", "range"}:
        raise AnnotationError(f"Unsupported axis_count suffix: {suffix!r}")
    if not axis:
        raise AnnotationError("@sy.axis_count annotation missing axis identifier")
    if not block_expr:
        block_expr = "1"
    return axis, kind, block_expr


def _parse_pid_map(comment: str) -> Dict[str, str]:
    payload = comment.split("@sy.pid_map", 1)[1].strip()
    parts = payload.split()
    result: Dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            raise AnnotationError(f"Invalid pid_map entry: {part!r}")
        axis, value = part.split("=", 1)
        result[axis.strip()] = value.strip()
    return result


def parse_annotations(lines: Iterable[str]) -> ParsedAnnotations:
    tile_id_var: Optional[str] = None
    tile_id_expr: Optional[str] = None
    axis_counts: List[AxisCount] = []
    pid_map: Dict[str, str] = {}
    dispatch_start: Optional[int] = None
    dispatch_end: Optional[int] = None
    dispatch_indent = ""
    persistent = False
    persistent_init_start: Optional[int] = None
    persistent_init_end: Optional[int] = None
    persistent_init_indent = ""
    producer_epilogue_idx: Optional[int] = None
    producer_epilogue_indent = ""

    for idx, line in enumerate(lines):
        if "@sy.tile_id" in line:
            lhs, rhs = _parse_assignment(_strip_comment(line))
            tile_id_var = lhs
            tile_id_expr = rhs
            if "persistent" in line:
                persistent = True
        if "@sy.axis_count" in line:
            axis, kind, block_expr = _parse_axis_count(line)
            var, expr = _parse_assignment(_strip_comment(line))
            axis_counts.append(
                AxisCount(axis=axis, var=var, kind=kind, expr=expr, block=block_expr)
            )
        if "@sy.pid_map" in line:
            pid_map.update(_parse_pid_map(line))
        if "@sy.dispatch begin" in line:
            dispatch_start = idx
            dispatch_indent = line[: len(line) - len(line.lstrip())]
        if "@sy.dispatch end" in line:
            dispatch_end = idx
        if "@sy.persistent_init begin" in line:
            persistent_init_start = idx
            persistent_init_indent = line[: len(line) - len(line.lstrip())]
        if "@sy.persistent_init end" in line:
            persistent_init_end = idx
        if "@sy.producer_epilogue" in line:
            if producer_epilogue_idx is not None:
                raise AnnotationError("Multiple @sy.producer_epilogue annotations found")
            producer_epilogue_idx = idx
            producer_epilogue_indent = line[: len(line) - len(line.lstrip())]

    if tile_id_var is None or tile_id_expr is None:
        raise AnnotationError("Missing @sy.tile_id annotation")
    if dispatch_start is None or dispatch_end is None:
        raise AnnotationError("Missing @sy.dispatch begin/end annotations")
    if dispatch_start >= dispatch_end:
        raise AnnotationError("@sy.dispatch begin must precede end")
    if not axis_counts:
        raise AnnotationError("At least one @sy.axis_count annotation is required")
    if not pid_map:
        raise AnnotationError("Missing @sy.pid_map annotation")
    if persistent and (persistent_init_start is None or persistent_init_end is None):
        raise AnnotationError(
            "Persistent kernels must mark initialization with @sy.persistent_init begin/end"
        )

    return ParsedAnnotations(
        tile_id_var=tile_id_var,
        tile_id_expr=tile_id_expr,
        axis_counts=axis_counts,
        pid_map=pid_map,
        dispatch=DispatchBlock(
            start=dispatch_start,
            end=dispatch_end,
            indent=dispatch_indent,
        ),
        persistent=persistent,
        persistent_init=None
        if persistent_init_start is None
        else DispatchBlock(
            start=persistent_init_start,
            end=persistent_init_end,  # type: ignore[arg-type]
            indent=persistent_init_indent,
        ),
        producer_epilogue=None
        if producer_epilogue_idx is None
        else DispatchBlock(
            start=producer_epilogue_idx,
            end=producer_epilogue_idx,
            indent=producer_epilogue_indent,
        ),
    )
