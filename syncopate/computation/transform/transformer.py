from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from syncopate.interface.tile_schedule import RawSchedule

from .annotations import AnnotationError, AxisCount, ParsedAnnotations, parse_annotations


@dataclass(frozen=True)
class RemoteBinding:
    """Binds a kernel-local descriptor to a base pointer + schedule.

    The schedule is consulted at transform time to decide:
      * whether the producer epilogue should use ``store`` or ``atomic_add``
        (driven by ``schedule.is_reduce()``)
      * whether compute reads or writes the descriptor
        (driven by ``schedule.role()``)

    Per-wave peer ranks come from ``schedule.gen_target_rank_list()`` at host
    setup time; the kernel simply receives the resulting tensor as
    ``target_rank``/``target_rank_<desc>``.
    """

    base_ptr_arg: str
    schedule: RawSchedule


@dataclass(frozen=True)
class _RemoteSpec:
    """Internal companion of RemoteBinding with derived names + decl range."""

    descriptor: str
    base_ptr_arg: str
    target_rank_arg: str
    cur_target_rank_var: str
    remote_ptr_var: str
    decl_start: int
    decl_end: int
    is_reduce: bool
    role: Optional[str]
    rebuild_template: List[str]  # absolute-indented lines for rebuild snippet
    dst_offsets_arg: str  # kernel arg name carrying per-wave dst offsets
    # axis_name -> (cur_dst_offset_var, dst_offset_adjust_var)
    axis_var_names: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    # For each descriptor offset position, the annotation-axis name it maps to
    # (or None when the descriptor axis is not partitioned by the schedule, so
    # call-site offsets at that position need no adjustment).
    descriptor_axes: Tuple[Optional[str], ...] = ()


def _find_def_start(lines: List[str], dispatch_start: int) -> int:
    for idx in range(dispatch_start, -1, -1):
        if lines[idx].lstrip().startswith("def "):
            return idx
    raise AnnotationError("Unable to locate function definition for annotations")


def _wave_symbol(axis: str, suffix: str) -> str:
    lower = axis.lower()
    return f"wave_{lower}_{suffix}"


def _current_tiles_var(axis: str) -> str:
    return f"cur_{axis.lower()}_tiles"


def _current_offset_var(axis: str) -> str:
    return f"cur_{axis.lower()}_offset"


@dataclass(frozen=True)
class _PersistentAxisInfo:
    axis: AxisCount
    index: int
    block_expr: str
    size_var: str
    offset_var: str


@dataclass(frozen=True)
class _SignalRole:
    kind: str  # "consumer" or "producer"
    ptr_arg: str
    offsets_arg: str
    persistent_offset_var: str
    nonpersistent_offset_var: str

    def offset_var(self, persistent: bool) -> str:
        return self.persistent_offset_var if persistent else self.nonpersistent_offset_var


@dataclass(frozen=True)
class _DispatchGenerationResult:
    lines: List[str]
    tile_count_var: Optional[str]
    wave_index_var: str
    signal_vars: Dict[str, str]


def _build_persistent_axis_info(parsed: ParsedAnnotations) -> List[_PersistentAxisInfo]:
    result: List[_PersistentAxisInfo] = []
    for idx, axis in enumerate(parsed.axis_counts):
        result.append(
            _PersistentAxisInfo(
                axis=axis,
                index=idx,
                block_expr=axis.block,
                size_var=_current_tiles_var(axis.axis),
                offset_var=_current_offset_var(axis.axis),
            )
        )
    return result


def _formatted_block_expr(expr: str) -> Optional[str]:
    text = expr.strip()
    if not text or text == "1":
        return None
    return text


def _scale_expression(expr: str, scale_expr: Optional[str]) -> str:
    if not scale_expr:
        return expr
    return f"({expr}) // ({scale_expr})"


def _block_product_expr(axis_counts: List[AxisCount]) -> Optional[str]:
    terms = []
    for axis in axis_counts:
        # reduction axes do not contribute to block size
        if axis.kind == "range":
            continue
        formatted = _formatted_block_expr(axis.block)
        if formatted:
            terms.append(f"({formatted})")
    if not terms:
        return None
    if len(terms) == 1:
        return terms[0]
    return " * ".join(terms)


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _find_descriptor_decl(lines: List[str], desc_name: str) -> Tuple[int, int]:
    """Locate the multi/single-line `<desc> = tl.make_tensor_descriptor(...)` block.

    Returns the inclusive range ``(start, end)``. Raises if not found or if the
    parenthesis tracking fails to close.
    """
    pattern = re.compile(
        rf"^\s*{re.escape(desc_name)}\s*=\s*tl\.make_tensor_descriptor\s*\("
    )
    for idx, line in enumerate(lines):
        if pattern.match(line):
            depth = line.count("(") - line.count(")")
            end = idx
            while depth > 0 and end < len(lines) - 1:
                end += 1
                depth += lines[end].count("(") - lines[end].count(")")
            if depth != 0:
                raise AnnotationError(
                    f"Unbalanced parentheses while parsing descriptor '{desc_name}'"
                )
            return idx, end
    raise AnnotationError(f"Descriptor declaration '{desc_name} = tl.make_tensor_descriptor(...)' not found")


def _replace_first_identifier(lines: List[str], identifier: str, replacement: str) -> List[str]:
    """Replace the first standalone occurrence of ``identifier`` across the lines."""
    pattern = re.compile(rf"\b{re.escape(identifier)}\b")
    for idx, line in enumerate(lines):
        match = pattern.search(line)
        if match is None:
            continue
        return lines[:idx] + [line[: match.start()] + replacement + line[match.end():]] + lines[idx + 1:]
    raise AnnotationError(f"Identifier '{identifier}' not found while rebuilding descriptor")


def _strip_indent(lines: List[str]) -> Tuple[str, List[str]]:
    """Return the common leading whitespace and the lines with that prefix stripped."""
    if not lines:
        return "", []
    base_indent: Optional[str] = None
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = line[: len(line) - len(stripped)]
        if base_indent is None or len(indent) < len(base_indent):
            base_indent = indent
    if base_indent is None:
        return "", list(lines)
    return base_indent, [line[len(base_indent):] if line.startswith(base_indent) else line for line in lines]


def _reindent(lines: List[str], indent: str) -> List[str]:
    return [f"{indent}{line}" if line.strip() else line for line in lines]


def _rewrite_store_to_atomic_add(lines: List[str], descriptor: str) -> None:
    """In-place: rewrite `<descriptor>.store(` to `<descriptor>.atomic_add(`."""
    pattern = re.compile(rf"\b{re.escape(descriptor)}\.store\(")
    replacement = f"{descriptor}.atomic_add("
    for idx, line in enumerate(lines):
        if pattern.search(line):
            lines[idx] = pattern.sub(replacement, line)


def _extract_descriptor_shape_axes(
    decl_lines: List[str], axis_names: List[str]
) -> Tuple[Optional[str], ...]:
    """Map each shape position of a `tl.make_tensor_descriptor(...)` declaration
    to the annotation axis whose name appears as an identifier at that position
    (or ``None`` if no annotation axis matches there)."""
    text = "\n".join(decl_lines)
    match = re.search(r"shape\s*=\s*\[([\s\S]*?)\]", text)
    if not match:
        return ()
    shape_str = match.group(1)
    positions = _split_top_level_commas(shape_str)
    result: List[Optional[str]] = []
    axis_set = set(axis_names)
    for part in positions:
        identifiers = set(_IDENTIFIER_RE.findall(part))
        hits = [name for name in axis_names if name in identifiers and name in axis_set]
        result.append(hits[0] if len(hits) == 1 else None)
    return tuple(result)


def _rewrite_descriptor_shape_positions(
    lines: List[str], peer_extents: List[Optional[str]]
) -> List[str]:
    """Replace matched positions in the `shape=[...]` of a
    ``make_tensor_descriptor`` declaration with peer-side extent expressions.

    ``peer_extents[i]`` is the textual replacement for position ``i`` (e.g.
    "4096") or ``None`` to leave that position untouched. The descriptor block
    is reconstructed line-by-line so original indentation is preserved when the
    shape spans multiple lines.
    """
    if not peer_extents or all(extent is None for extent in peer_extents):
        return lines
    text = "\n".join(lines)
    match = re.search(r"(shape\s*=\s*\[)([\s\S]*?)(\])", text)
    if not match:
        return lines
    parts = _split_top_level_commas(match.group(2))
    if len(parts) != len(peer_extents):
        return lines
    rewritten: List[str] = []
    for part, extent in zip(parts, peer_extents):
        if extent is None:
            rewritten.append(part)
        else:
            leading = part[: len(part) - len(part.lstrip())]
            trailing = part[len(part.rstrip()) :]
            rewritten.append(f"{leading}{extent}{trailing}")
    new_shape = ",".join(rewritten)
    new_text = text[: match.start(2)] + new_shape + text[match.end(2) :]
    return new_text.split("\n")


def _split_top_level_commas(text: str) -> List[str]:
    """Split a string by top-level commas (ignoring those inside parens/brackets)."""
    parts: List[str] = []
    depth = 0
    last = 0
    for idx, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[last:idx])
            last = idx + 1
    parts.append(text[last:])
    return parts


def _rewrite_descriptor_call_offsets(
    lines: List[str],
    descriptor: str,
    adjust_exprs: List[Optional[str]],
) -> None:
    """Add per-axis adjustment expressions to each offset element of every
    `<descriptor>.<op>([offs0, offs1, ...], ...)` call site, in-place.

    ``adjust_exprs[i]`` is the expression added to the i-th offset element; if
    ``None``, that axis is left untouched.
    """
    if not any(expr for expr in adjust_exprs):
        return
    pattern = re.compile(
        rf"\b{re.escape(descriptor)}\.\w+\s*\(\s*\["
    )
    for idx, line in enumerate(lines):
        match = pattern.search(line)
        if match is None:
            continue
        start = match.end()  # position right after the opening '['
        depth = 1
        end = start
        while end < len(line) and depth > 0:
            ch = line[end]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    break
            end += 1
        if depth != 0:
            # offset list spans multiple lines — skip rather than corrupt
            continue
        offsets_str = line[start:end]
        parts = _split_top_level_commas(offsets_str)
        if len(parts) != len(adjust_exprs):
            continue
        rewritten: List[str] = []
        for raw, adjust in zip(parts, adjust_exprs):
            stripped = raw.strip()
            if not stripped:
                rewritten.append(raw)
                continue
            if adjust is None:
                rewritten.append(raw)
            else:
                leading = raw[: len(raw) - len(raw.lstrip())]
                trailing = raw[len(raw.rstrip()):]
                rewritten.append(f"{leading}({stripped}) + {adjust}{trailing}")
        new_offsets = ",".join(rewritten)
        lines[idx] = line[:start] + new_offsets + line[end:]


def _nonrange_block_exprs(axis_counts: List[AxisCount]) -> List[str]:
    blocks: List[str] = []
    for axis in axis_counts:
        if axis.kind == "range":
            continue
        formatted = _formatted_block_expr(axis.block)
        if formatted:
            blocks.append(formatted)
    return blocks


def _format_host_block_expr(expr: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.isupper():
            return f'META["{token}"]'
        return token

    return _IDENTIFIER_RE.sub(repl, expr)


def _host_tile_shape_expr(axis_counts: List[AxisCount]) -> str:
    exprs: List[str] = []
    for block in _nonrange_block_exprs(axis_counts):
        formatted = _format_host_block_expr(block)
        if formatted and formatted != "1":
            exprs.append(f"({formatted})")
    if not exprs:
        return "1"
    return " * ".join(exprs)


def _kernel_tile_shape_expr(axis_counts: List[AxisCount]) -> str:
    exprs: List[str] = []
    for block in _nonrange_block_exprs(axis_counts):
        exprs.append(f"({block})")
    if not exprs:
        return "1"
    if len(exprs) == 1:
        return exprs[0]
    return " * ".join(exprs)


def _word_boundary_replace(text: str, replacements: Dict[str, str]) -> str:
    if not replacements:
        return text
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in replacements) + r")\b")
    return pattern.sub(lambda m: replacements[m.group(0)], text)


def _extract_assigned_pids(line: str, pid_vars: List[str]) -> List[str]:
    if "=" not in line:
        return []
    lhs, _, _ = line.partition("=")
    names = [name.strip() for name in lhs.split(",")]
    return [name for name in names if name in pid_vars]


def _parse_pid_target(target: str) -> Tuple[str, Tuple[str, ...]]:
    cleaned = target.strip()
    if not cleaned:
        raise AnnotationError("Empty pid_map target")
    if ".." in cleaned:
        start, end = cleaned.split("..", 1)
        start = start.strip()
        end = end.strip()
        if not start or not end:
            raise AnnotationError(f"Invalid range pid_map entry: {target!r}")
        return "range", (start, end)
    return "single", (cleaned,)


def _ensure_import(lines: List[str], statement: str) -> List[str]:
    if any(statement in line for line in lines):
        return lines

    insert_idx = 0
    if lines:
        stripped = lines[0].lstrip()
        if stripped.startswith(("'''", '"""')):
            quote = stripped[:3]
            idx = 1
            while idx < len(lines):
                segment = lines[idx]
                if quote in segment:
                    if segment.count(quote) % 2 == 1:
                        idx += 1
                        break
                idx += 1
            insert_idx = idx

    while insert_idx < len(lines) and not lines[insert_idx].strip():
        insert_idx += 1

    last_import = insert_idx - 1
    for idx in range(insert_idx, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import = idx
            continue
        if not stripped:
            break
        break

    if last_import >= insert_idx:
        insert_pos = last_import + 1
    else:
        insert_pos = insert_idx

    return lines[:insert_pos] + [statement] + lines[insert_pos:]


def _augment_signature_lines(
    lines: List[str], def_start: int, extra_args: str
) -> Tuple[List[str], int]:
    guard_tokens: List[str] = []
    for part in extra_args.split(","):
        token = part.strip()
        if not token:
            continue
        token = token.split("=", 1)[0]
        token = token.split(":", 1)[0].strip()
        if token:
            guard_tokens.append(token)

    line = lines[def_start]
    if "(" in line and ")" in line:
        if any(token and token in line for token in guard_tokens):
            return [line], def_start
        indent = line[: len(line) - len(line.lstrip())]
        stripped = line.strip()
        head, remainder = stripped.split("(", 1)
        args_part, suffix = remainder.rsplit(")", 1)
        args_part = args_part.strip()
        extra = extra_args
        if args_part:
            args_part = f"{args_part}, {extra}"
        else:
            args_part = extra
        return [f"{indent}{head}({args_part}){suffix}"], def_start

    depth = 0
    def_end = def_start
    seen_paren = False
    for idx in range(def_start, len(lines)):
        text = lines[idx]
        if "(" in text:
            depth += text.count("(")
            seen_paren = True
        if ")" in text:
            depth -= text.count(")")
        if seen_paren and depth <= 0:
            def_end = idx
            break
    else:
        raise AnnotationError("Unable to determine end of function definition")

    def_lines = lines[def_start : def_end + 1]
    signature_text = "\n".join(def_lines)
    if any(token and token in signature_text for token in guard_tokens):
        return def_lines, def_end

    arg_indent = None
    for idx in range(def_start + 1, def_end + 1):
        stripped = lines[idx].lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        arg_indent = lines[idx][: len(lines[idx]) - len(stripped)]
        break
    if arg_indent is None:
        arg_indent = "    "

    prev_arg_idx = None
    for idx in range(def_end - 1, def_start, -1):
        stripped = lines[idx].strip()
        if not stripped or stripped.startswith("#"):
            continue
        prev_arg_idx = idx - def_start
        break

    def_lines = def_lines.copy()
    if prev_arg_idx is not None:
        original_line = def_lines[prev_arg_idx]
        head, sep, comment = original_line.partition("#")
        head = head.rstrip()
        if head and not head.endswith(","):
            head += ","
        new_line = head
        if sep:
            new_line += f" #{comment}"
        def_lines[prev_arg_idx] = new_line

    extra_line = f"{arg_indent}{extra_args}"
    if not extra_line.rstrip().endswith(","):
        extra_line = extra_line + ","
    def_lines.insert(len(def_lines) - 1, extra_line)
    return def_lines, def_end


def _generate_dispatch_block(
    lines: List[str],
    parsed: ParsedAnnotations,
    *,
    signal_roles: Tuple[_SignalRole, ...],
    consumer_role: Optional[_SignalRole],
    consumer_descriptors: Tuple[str, ...],
    enable_producer: bool,
    remote_specs: Tuple[_RemoteSpec, ...] = (),
) -> _DispatchGenerationResult:
    axis_counts = parsed.axis_counts
    num_axes = len(axis_counts)
    indent = parsed.dispatch.indent
    block_product = _block_product_expr(axis_counts)

    block: List[str] = []
    block.append(f"{indent}# auto-generated dispatch (wave-based)")
    block.append(f"{indent}wave_dim: tl.constexpr = {num_axes}")

    tile_var = parsed.tile_id_var
    local_tile_var = f"local_{tile_var}"
    block.append(f"{indent}cum_wave_range = tl.arange(0, NUM_WAVES)")
    block.append(f"{indent}cum_wave_sizes_vec = tl.load(cum_wave_sizes + cum_wave_range)")
    if block_product:
        block.append(
            f"{indent}cum_wave_sizes_vec = {_scale_expression('cum_wave_sizes_vec', block_product)}"
        )
    block.append(
        f"{indent}wave_candidates = tl.where(cum_wave_sizes_vec > {tile_var}, cum_wave_range, NUM_WAVES)"
    )
    block.append(f"{indent}wave_idx = tl.min(wave_candidates)")
    for role in signal_roles:
        offset_var = role.nonpersistent_offset_var
        block.append(
            f"{indent}{offset_var} = tl.load({role.offsets_arg} + wave_idx)"
        )
    if consumer_role is not None:
        offset_var = consumer_role.nonpersistent_offset_var
        ptr_arg = consumer_role.ptr_arg
        inner = indent + "    "
        block.append(f"{indent}if {offset_var} >= 0:")
        block.append(
            f"{inner}token = dl.wait({ptr_arg} + {offset_var}, 1, \"gpu\", \"acquire\", waitValue=1)"
        )
        for descriptor in consumer_descriptors:
            block.append(f"{inner}{descriptor} = dl.consume_token({descriptor}, token)")
    for spec in remote_specs:
        block.append(
            f"{indent}{spec.cur_target_rank_var} = tl.load({spec.target_rank_arg} + wave_idx)"
        )
        block.append(
            f"{indent}{spec.remote_ptr_var} = dl.symm_at({spec.base_ptr_arg}, {spec.cur_target_rank_var})"
        )
        for line in _reindent(spec.rebuild_template, indent):
            block.append(line)
    previous_load = "tl.load(cum_wave_sizes + wave_idx - 1)"
    if block_product:
        previous_load = _scale_expression(previous_load, block_product)
    block.append(
        f"{indent}previous_cum = tl.where(wave_idx == 0, 0, {previous_load})"
    )
    block.append(f"{indent}{local_tile_var} = {tile_var} - previous_cum")

    replacements: Dict[str, str] = {tile_var: local_tile_var}
    axis_info: List[Tuple[AxisCount, str, str, str]] = []
    pid_offsets: Dict[str, str] = {}

    for axis_index, axis in enumerate(axis_counts):
        offset_name = _wave_symbol(axis.axis, "offset")
        size_name = _wave_symbol(axis.axis, "size")
        local_axis_var = f"local_{axis.var}"
        pid_var = parsed.pid_map.get(axis.axis)
        if pid_var is None:
            raise AnnotationError(f"Missing pid mapping for axis '{axis.axis}'")
        pid_kind, pid_targets = _parse_pid_target(pid_var)
        axis_kind = "range" if axis.kind == "range" or pid_kind == "range" else "single"
        offset_expr = f"tl.load(wave_offsets + wave_idx * wave_dim + {axis_index})"
        size_expr = f"tl.load(wave_sizes + wave_idx * wave_dim + {axis_index})"
        axis_block = _formatted_block_expr(axis.block)
        if axis_block:
            offset_expr = _scale_expression(offset_expr, axis_block)
            size_expr = _scale_expression(size_expr, axis_block)
        block.append(f"{indent}{offset_name} = {offset_expr}")
        block.append(f"{indent}{size_name} = {size_expr}")
        block.append(f"{indent}{local_axis_var} = {size_name}")
        replacements[axis.var] = local_axis_var
        axis_info.append((axis, offset_name, size_name, local_axis_var))
        if axis_kind == "range":
            if len(pid_targets) != 2:
                raise AnnotationError(
                    f"Range axis '{axis.axis}' must map to two pid variables"
                )
            start_var, end_var = pid_targets
            if "(" not in start_var and ")" not in start_var:
                pid_offsets[start_var] = offset_name
            if "(" not in end_var and ")" not in end_var:
                pid_offsets[end_var] = offset_name
        else:
            single_pid = pid_targets[0]
            if "(" not in single_pid and ")" not in single_pid:
                pid_offsets[single_pid] = offset_name

    body_lines = lines[parsed.dispatch.start + 1 : parsed.dispatch.end]
    offset_inserted = set()

    for original_line in body_lines:
        if "@sy." in original_line:
            continue
        raw = original_line.rstrip()
        stripped = raw.split("#", 1)[0].strip()
        if not stripped:
            block.append("")
            continue
        replaced = _word_boundary_replace(raw, replacements)
        block.append(replaced)
        assigned = _extract_assigned_pids(stripped, list(pid_offsets.keys()))
        for pid_var in assigned:
            if pid_var not in offset_inserted:
                block.append(f"{indent}{pid_var} += {pid_offsets[pid_var]}")
                offset_inserted.add(pid_var)

    tile_count_var = None
    if enable_producer:
        # don't count the reduction axes
        tiles_expr = " * ".join(
            local_var for axis, _, _, local_var in axis_info if axis.kind != "range"
        ) or "1"
        tile_count_var = "_sy_wave_tile_count"
        block.append(f"{indent}{tile_count_var} = {tiles_expr}")

    context = _DispatchGenerationResult(
        lines=block,
        tile_count_var=tile_count_var,
        wave_index_var="wave_idx",
        signal_vars={role.kind: role.nonpersistent_offset_var for role in signal_roles},
    )
    return context


def _emit_remote_dst_offset_setup(
    indent: str,
    remote_specs: Tuple[_RemoteSpec, ...],
    axis_info: List[_PersistentAxisInfo],
    *,
    wave_dim_name: str,
    wave_var: str,
) -> List[str]:
    """Emit per-binding `cur_<desc>_dst_offset_<axis>` loads + adjustment vars.

    For each remote binding, loads the per-wave dst offset for each axis (in
    tile units, divided by BLOCK_SIZE_<axis>) and pre-computes the byte-level
    adjustment expression `(cur_dst_offset - cur_offset) * BLOCK_SIZE` once per
    wave so call-site rewrites can be a single `+ <adjust>` per offset.
    """
    out: List[str] = []
    for spec in remote_specs:
        for info in axis_info:
            cur_var, _ = spec.axis_var_names[info.axis.axis]
            block_expr = _formatted_block_expr(info.block_expr)
            load_expr = f"tl.load({spec.dst_offsets_arg} + {wave_var} * {wave_dim_name} + {info.index})"
            load_expr = _scale_expression(load_expr, block_expr)
            out.append(f"{indent}{cur_var} = {load_expr}")
        for info in axis_info:
            cur_var, adjust_var = spec.axis_var_names[info.axis.axis]
            block_expr = _formatted_block_expr(info.block_expr)
            mul = f" * ({block_expr})" if block_expr else ""
            out.append(
                f"{indent}{adjust_var} = ({cur_var} - {info.offset_var}){mul}"
            )
    return out


def _generate_persistent_init_block(
    parsed: ParsedAnnotations,
    axis_info: List[_PersistentAxisInfo],
    *,
    signal_roles: Tuple[_SignalRole, ...],
    wave_dim_name: str,
    cur_wave_sizes_arg: str,
    wave_offsets_arg: str,
    remote_specs: Tuple[_RemoteSpec, ...] = (),
) -> List[str]:
    if parsed.persistent_init is None:
        raise AnnotationError("Persistent kernels require @sy.persistent_init region")
    indent = parsed.persistent_init.indent
    block: List[str] = []
    block.append(f"{indent}# auto-generated persistent init (wave-based)")
    block.append(f"{indent}{wave_dim_name}: tl.constexpr = {len(axis_info)}")
    block.append(f"{indent}cur_wave = 0")
    block.append(f"{indent}new_wave = True")
    for info in axis_info:
        load_expr = f"tl.load({wave_offsets_arg} + cur_wave * {wave_dim_name} + {info.index})"
        load_expr = _scale_expression(load_expr, _formatted_block_expr(info.block_expr))
        block.append(f"{indent}{info.offset_var} = {load_expr}")
    block.append("")
    block.append(f"{indent}past_tiles = 0")
    for info in axis_info:
        size_expr = f"tl.load({cur_wave_sizes_arg} + cur_wave * {wave_dim_name} + {info.index})"
        size_expr = _scale_expression(size_expr, _formatted_block_expr(info.block_expr))
        block.append(f"{indent}{info.size_var} = {size_expr}")
    for role in signal_roles:
        block.append(
            f"{indent}{role.persistent_offset_var} = tl.load({role.offsets_arg} + cur_wave)"
        )
    filtered = [info.size_var for info in axis_info if info.axis.kind != "range"]
    tiles_expr = " * ".join(filtered) or "1"
    block.append(f"{indent}cur_tiles = {tiles_expr}")
    block.extend(
        _emit_remote_dst_offset_setup(
            indent,
            remote_specs,
            axis_info,
            wave_dim_name=wave_dim_name,
            wave_var="cur_wave",
        )
    )
    return block


def _generate_persistent_dispatch_block(
    lines: List[str],
    parsed: ParsedAnnotations,
    axis_info: List[_PersistentAxisInfo],
    *,
    signal_roles: Tuple[_SignalRole, ...],
    consumer_role: Optional[_SignalRole],
    consumer_descriptors: Tuple[str, ...],
    wave_dim_name: str,
    cur_wave_sizes_arg: str,
    wave_offsets_arg: str,
    remote_specs: Tuple[_RemoteSpec, ...] = (),
) -> _DispatchGenerationResult:
    indent = parsed.dispatch.indent
    block: List[str] = []
    block.append(f"{indent}# auto-generated persistent dispatch (wave-based)")
    while_indent = indent + "    "
    block.append(f"{indent}while {parsed.tile_id_var} >= past_tiles + cur_tiles:")
    block.append(f"{while_indent}past_tiles += cur_tiles")
    block.append(f"{while_indent}cur_wave += 1")
    for info in axis_info:
        size_expr = f"tl.load({cur_wave_sizes_arg} + cur_wave * {wave_dim_name} + {info.index})"
        size_expr = _scale_expression(size_expr, _formatted_block_expr(info.block_expr))
        block.append(f"{while_indent}{info.size_var} = {size_expr}")
    for info in axis_info:
        offset_expr = f"tl.load({wave_offsets_arg} + cur_wave * {wave_dim_name} + {info.index})"
        offset_expr = _scale_expression(offset_expr, _formatted_block_expr(info.block_expr))
        block.append(f"{while_indent}{info.offset_var} = {offset_expr}")
    # if not reduction axis
    filtered = [info.size_var for info in axis_info if info.axis.kind != "range"]
    tiles_expr = " * ".join(filtered) or "1"
    block.append(f"{while_indent}cur_tiles = {tiles_expr}")
    for role in signal_roles:
        block.append(
            f"{while_indent}{role.persistent_offset_var} = tl.load({role.offsets_arg} + cur_wave)"
        )
    block.append(f"{while_indent}new_wave = True")
    if consumer_role is not None or remote_specs:
        block.append(f"{indent}if new_wave:")
        new_wave_indent = indent + "    "
        block.append(f"{new_wave_indent}new_wave = False")
        if consumer_role is not None:
            offset_var = consumer_role.persistent_offset_var
            ptr_arg = consumer_role.ptr_arg
            block.append(f"{new_wave_indent}if {offset_var} >= 0:")
            wait_indent = new_wave_indent + "    "
            block.append(
                f"{wait_indent}token = dl.wait({ptr_arg} + {offset_var}, 1, \"gpu\", \"acquire\", waitValue=1)"
            )
            for descriptor in consumer_descriptors:
                block.append(f"{wait_indent}{descriptor} = dl.consume_token({descriptor}, token)")
        for spec in remote_specs:
            block.append(
                f"{new_wave_indent}{spec.cur_target_rank_var} = tl.load({spec.target_rank_arg} + cur_wave)"
            )
            block.append(
                f"{new_wave_indent}{spec.remote_ptr_var} = dl.symm_at({spec.base_ptr_arg}, {spec.cur_target_rank_var})"
            )
            for line in _reindent(spec.rebuild_template, new_wave_indent):
                block.append(line)
        block.extend(
            _emit_remote_dst_offset_setup(
                new_wave_indent,
                remote_specs,
                axis_info,
                wave_dim_name=wave_dim_name,
                wave_var="cur_wave",
            )
        )

    block.append("")
    tile_var = parsed.tile_id_var
    replacements: Dict[str, str] = {tile_var: f"{tile_var} - past_tiles"}
    pid_offsets: Dict[str, str] = {}
    for info in axis_info:
        axis = info.axis
        replacements[axis.var] = info.size_var
        pid_var = parsed.pid_map.get(axis.axis)
        if pid_var is None:
            raise AnnotationError(f"Missing pid mapping for axis '{axis.axis}'")
        pid_kind, pid_targets = _parse_pid_target(pid_var)
        axis_kind = "range" if axis.kind == "range" or pid_kind == "range" else "single"
        if axis_kind == "range":
            if len(pid_targets) != 2:
                raise AnnotationError(
                    f"Range axis '{axis.axis}' must map to two pid variables"
                )
            start_var, end_var = pid_targets
            if "(" not in start_var and ")" not in start_var:
                pid_offsets[start_var] = info.offset_var
            if "(" not in end_var and ")" not in end_var:
                pid_offsets[end_var] = info.offset_var
        else:
            single_pid = pid_targets[0]
            if "(" not in single_pid and ")" not in single_pid:
                pid_offsets[single_pid] = info.offset_var

    body_lines = lines[parsed.dispatch.start + 1 : parsed.dispatch.end]
    offset_inserted = set()
    for original_line in body_lines:
        if "@sy." in original_line:
            continue
        raw = original_line.rstrip()
        stripped = raw.split("#", 1)[0].strip()
        if not stripped:
            block.append("")
            continue
        replaced = _word_boundary_replace(raw, replacements)
        block.append(replaced)
        assigned = _extract_assigned_pids(stripped, list(pid_offsets.keys()))
        for pid_var in assigned:
            if pid_var not in offset_inserted:
                block.append(f"{indent}{pid_var} += {pid_offsets[pid_var]}")
                offset_inserted.add(pid_var)
    context = _DispatchGenerationResult(
        lines=block,
        tile_count_var="cur_tiles",
        wave_index_var="cur_wave",
        signal_vars={role.kind: role.persistent_offset_var for role in signal_roles},
    )
    return context


def _generate_producer_epilogue_block(
    indent: str,
    *,
    signal_ptr: str,
    signal_offset_var: str,
    counter_ptr: str,
    tile_count_var: str,
    wave_index_var: str,
) -> List[str]:
    block: List[str] = []
    block.append(f"{indent}# auto-generated producer epilogue (wave-based)")
    block.append(f"{indent}if {signal_offset_var} >= 0:")
    inner = indent + "    "
    block.append(
        f"{inner}val = tl.atomic_add({counter_ptr} + {wave_index_var}, 1, sem=\"release\", scope=\"gpu\")"
    )
    block.append(f"{inner}if val == {tile_count_var} - 1:")
    notify_indent = inner + "    "
    block.append(
        f"{notify_indent}dl.notify({signal_ptr} + {signal_offset_var}, dl.rank(), signal=1, comm_scope=\"gpu\")"
    )
    return block


def _add_kwargs_to_host_signature(lines: List[str], def_idx: int) -> None:
    joined = "".join(lines[def_idx : def_idx + 1])
    if "**kwargs" in joined:
        return
    for idx in range(def_idx, len(lines)):
        line = lines[idx]
        if "):" not in line:
            continue
        close_idx = line.find("):")
        if close_idx == -1:
            continue
        if "**kwargs" in line[:close_idx]:
            return
        lines[idx] = line[:close_idx] + ", **kwargs" + line[close_idx:]
        return


def _find_function_end(lines: List[str], def_idx: int) -> int:
    def_indent = len(lines[def_idx]) - len(lines[def_idx].lstrip())
    for idx in range(def_idx + 1, len(lines)):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        current_indent = len(lines[idx]) - len(lines[idx].lstrip())
        if current_indent <= def_indent:
            return idx
    return len(lines)


def _rewrite_host_num_tiles(
    lines: List[str],
    start: int,
    end: int,
    *,
    tile_shape_expr: str,
) -> None:
    idx = start
    while idx < end:
        line = lines[idx]
        if "@sy.num_tiles" not in line:
            idx += 1
            continue
        if "cum_tiles" in line:
            idx += 1
            continue
        indent = line[: len(line) - len(line.lstrip())]
        lhs, _, remainder = line.partition("=")
        var_name = lhs.strip()
        rhs = remainder.split("#", 1)[0].strip()
        inner_indent = indent + "    "
        replacement = [
            f"{indent}tile_shape = {tile_shape_expr}",
            f"{indent}cum_tiles = kwargs.get(\"cum_tiles\")",
            f"{indent}if cum_tiles is None:",
            f"{inner_indent}cum_tiles = ({rhs}) * tile_shape",
            f"{indent}{var_name} = triton.cdiv(cum_tiles, tile_shape) # @sy.num_tiles",
        ]
        lines[idx : idx + 1] = replacement
        delta = len(replacement) - 1
        end += delta
        idx += len(replacement)
    return


def _inject_kwargs_into_kernel_launch(
    lines: List[str], start: int, end: int
) -> int:
    idx = start
    while idx < end:
        line = lines[idx]
        if "@sy.kernel_launch" not in line:
            idx += 1
            continue
        call_idx = idx + 1
        while call_idx < end and "(" not in lines[call_idx]:
            call_idx += 1
        if call_idx >= end:
            return end
        depth = lines[call_idx].count("(") - lines[call_idx].count(")")
        scan_idx = call_idx + 1
        while depth > 0 and scan_idx < len(lines):
            depth += lines[scan_idx].count("(") - lines[scan_idx].count(")")
            scan_idx += 1
        closing_idx = scan_idx - 1
        if any("**kwargs" in lines[pos] for pos in range(call_idx, closing_idx)):
            idx = closing_idx + 1
            continue
        arg_idx = closing_idx - 1
        while arg_idx > call_idx and not lines[arg_idx].strip():
            arg_idx -= 1
        indent = lines[arg_idx][: len(lines[arg_idx]) - len(lines[arg_idx].lstrip())]
        lines.insert(closing_idx, f"{indent}**kwargs,")
        end += 1
        idx = closing_idx + 1
    return end


def _rewrite_kernel_num_tiles(
    lines: List[str],
    axis_counts: List[AxisCount],
) -> None:
    tile_shape_expr = _kernel_tile_shape_expr(axis_counts)
    for idx, line in enumerate(lines):
        if "@sy.num_tiles" not in line:
            continue
        if "cum_tiles" in line:
            return
        indent = line[: len(line) - len(line.lstrip())]
        lhs, _, remainder = line.partition("=")
        var_name = lhs.strip()
        rhs = remainder.split("#", 1)[0].strip()
        inner_indent = indent + "    "
        lines_to_insert: List[str] = []
        if tile_shape_expr != "1":
            lines_to_insert.append(f"{indent}tile_shape = {tile_shape_expr}")
            cdiv_expr = "tl.cdiv(cum_tiles, tile_shape)"
        else:
            cdiv_expr = "cum_tiles"
        lines_to_insert.append(f"{indent}if cum_tiles != 0:")
        lines_to_insert.append(
            f"{inner_indent}{var_name} = {cdiv_expr} # @sy.num_tiles"
        )
        lines_to_insert.append(f"{indent}else:")
        lines_to_insert.append(f"{inner_indent}{var_name} = {rhs}")
        lines[idx : idx + 1] = lines_to_insert
        return
    return


def _transform_host_entries(
    lines: List[str],
    axis_counts: List[AxisCount],
) -> List[str]:
    idx = 0
    tile_shape_expr = _host_tile_shape_expr(axis_counts)
    while idx < len(lines):
        line = lines[idx]
        if "@sy.host_entry" not in line:
            idx += 1
            continue
        _add_kwargs_to_host_signature(lines, idx)
        func_end = _find_function_end(lines, idx)
        _rewrite_host_num_tiles(
            lines,
            idx + 1,
            func_end,
            tile_shape_expr=tile_shape_expr,
        )
        func_end = _inject_kwargs_into_kernel_launch(lines, idx + 1, func_end)
        idx = func_end
    return lines


class AnnotationTransformer:
    """Transforms annotated kernels into their wave-dispatch template form."""

    def __init__(
        self,
        *,
        enable_consumer: bool = False,
        consumer_descriptors: Tuple[str, ...] = ("desc_k", "desc_v"),
        enable_producer: bool = False,
        remote_descriptors: Optional[Mapping[str, RemoteBinding]] = None,
    ) -> None:
        if enable_consumer and not consumer_descriptors:
            raise ValueError(
                "consumer_descriptors must be provided when enabling consumer mode"
            )
        self.enable_consumer = enable_consumer
        self.enable_producer = enable_producer
        self.consumer_descriptors = tuple(consumer_descriptors)
        self.remote_descriptors: Dict[str, RemoteBinding] = dict(remote_descriptors or {})

        multi_role = int(enable_consumer) + int(enable_producer) > 1

        def _offset_var(base: str, kind: str) -> str:
            if not multi_role:
                return base
            return base.replace("signal_offset", f"{kind}_signal_offset")

        self.consumer_signal_role: Optional[_SignalRole] = None
        if enable_consumer:
            ptr_arg = "signal_ptr" if not enable_producer else "consumer_signal_ptr"
            offsets_arg = "signal_offsets" if not enable_producer else "consumer_signal_offsets"
            persistent_var = _offset_var("cur_signal_offset", "consumer")
            nonpersistent_var = _offset_var("wave_signal_offset", "consumer")
            self.consumer_signal_role = _SignalRole(
                kind="consumer",
                ptr_arg=ptr_arg,
                offsets_arg=offsets_arg,
                persistent_offset_var=persistent_var,
                nonpersistent_offset_var=nonpersistent_var,
            )

        self.producer_signal_role: Optional[_SignalRole] = None
        self.producer_counter_arg: Optional[str] = None
        if enable_producer:
            ptr_arg = "signal_ptr" if not enable_consumer else "producer_signal_ptr"
            offsets_arg = "signal_offsets" if not enable_consumer else "producer_signal_offsets"
            persistent_var = _offset_var("cur_signal_offset", "producer")
            nonpersistent_var = _offset_var("wave_signal_offset", "producer")
            self.producer_signal_role = _SignalRole(
                kind="producer",
                ptr_arg=ptr_arg,
                offsets_arg=offsets_arg,
                persistent_offset_var=persistent_var,
                nonpersistent_offset_var=nonpersistent_var,
            )
            self.producer_counter_arg = (
                "counter_ptr" if not enable_consumer else "producer_counter_ptr"
            )

        roles: List[_SignalRole] = []
        if self.consumer_signal_role is not None:
            roles.append(self.consumer_signal_role)
        if self.producer_signal_role is not None:
            roles.append(self.producer_signal_role)
        self.signal_roles = tuple(roles)

    def _nonpersistent_extra_args(self) -> str:
        args = [
            "wave_offsets=None",
            "wave_sizes=None",
            "cum_wave_sizes=None",
        ]
        args.extend(self._signal_extra_args())
        if self.enable_producer and self.producer_counter_arg is not None:
            args.append(f"{self.producer_counter_arg}=None")
        args.extend(self._remote_extra_args())
        args.append("cum_tiles=0")
        args.append("NUM_WAVES: tl.constexpr=0")
        return ", ".join(args)

    def _persistent_extra_args(self) -> str:
        args = [
            "cur_wave_sizes=None",
            "wave_offsets=None",
        ]
        args.extend(self._signal_extra_args())
        if self.enable_producer and self.producer_counter_arg is not None:
            args.append(f"{self.producer_counter_arg}=None")
        args.extend(self._remote_extra_args())
        args.append("cum_tiles=0")
        return ", ".join(args)

    def _signal_extra_args(self) -> List[str]:
        args: List[str] = []
        for role in self.signal_roles:
            args.append(f"{role.ptr_arg}=None")
            args.append(f"{role.offsets_arg}=None")
        return args

    def _remote_target_rank_arg(self, descriptor: str) -> str:
        """Kernel arg name that carries per-wave peer ranks for a given descriptor.

        Single-binding mode collapses to plain ``target_rank`` for ergonomics;
        multi-binding mode disambiguates by descriptor name.
        """
        if len(self.remote_descriptors) <= 1:
            return "target_rank"
        return f"target_rank_{descriptor}"

    def _remote_dst_offsets_arg(self, descriptor: str) -> str:
        if len(self.remote_descriptors) <= 1:
            return "dst_offsets"
        return f"dst_offsets_{descriptor}"

    def _remote_extra_args(self) -> List[str]:
        args: List[str] = []
        for name in self.remote_descriptors:
            args.append(f"{self._remote_target_rank_arg(name)}=None")
            args.append(f"{self._remote_dst_offsets_arg(name)}=None")
        return args

    def _build_remote_specs(
        self, lines: List[str], parsed: ParsedAnnotations
    ) -> Tuple[_RemoteSpec, ...]:
        if not self.remote_descriptors:
            return ()
        single = len(self.remote_descriptors) == 1
        specs: List[_RemoteSpec] = []
        for desc_name, binding in self.remote_descriptors.items():
            target_rank_arg = self._remote_target_rank_arg(desc_name)
            dst_offsets_arg = self._remote_dst_offsets_arg(desc_name)
            cur_target_rank_var = (
                "cur_target_rank" if single else f"cur_target_rank_{desc_name}"
            )
            remote_ptr_var = (
                f"remote_{binding.base_ptr_arg}"
                if single
                else f"remote_{binding.base_ptr_arg}_{desc_name}"
            )
            decl_start, decl_end = _find_descriptor_decl(lines, desc_name)
            decl_lines = lines[decl_start : decl_end + 1]
            base_indent_str, dedented = _strip_indent(decl_lines)
            rebuild_template = _replace_first_identifier(
                dedented, binding.base_ptr_arg, remote_ptr_var
            )
            descriptor_axes = _extract_descriptor_shape_axes(
                decl_lines, [axis.axis for axis in parsed.axis_counts]
            )

            # The descriptor lives on a peer's allocation (per-rank slice), so
            # its declared shape must match peer-side extents — not the global
            # tensor shape baked into the source. Per-axis extents come from
            # the schedule's first wave tile shape (assumed uniform across
            # waves; non-uniform schedules need a runtime-loaded variant).
            schedule_blocks = binding.schedule.block_infos
            first_block_shape: Tuple[int, ...] = ()
            if schedule_blocks and schedule_blocks[0]:
                first_block_shape = schedule_blocks[0][0].tile.shape
            peer_extents: List[Optional[str]] = []
            for pos_axis_name in descriptor_axes:
                if pos_axis_name is None:
                    peer_extents.append(None)
                    continue
                annot_idx = next(
                    (
                        i
                        for i, ax in enumerate(parsed.axis_counts)
                        if ax.axis == pos_axis_name
                    ),
                    None,
                )
                if annot_idx is None or annot_idx >= len(first_block_shape):
                    peer_extents.append(None)
                else:
                    peer_extents.append(str(int(first_block_shape[annot_idx])))
            rebuild_template = _rewrite_descriptor_shape_positions(
                rebuild_template, peer_extents
            )

            axis_var_names: Dict[str, Tuple[str, str]] = {}
            for axis in parsed.axis_counts:
                cur_var = (
                    f"cur_dst_offset_{axis.axis.lower()}"
                    if single
                    else f"cur_{desc_name}_dst_offset_{axis.axis.lower()}"
                )
                adjust_var = (
                    f"dst_offset_adjust_{axis.axis.lower()}"
                    if single
                    else f"{desc_name}_dst_offset_adjust_{axis.axis.lower()}"
                )
                axis_var_names[axis.axis] = (cur_var, adjust_var)

            specs.append(
                _RemoteSpec(
                    descriptor=desc_name,
                    base_ptr_arg=binding.base_ptr_arg,
                    target_rank_arg=target_rank_arg,
                    cur_target_rank_var=cur_target_rank_var,
                    remote_ptr_var=remote_ptr_var,
                    decl_start=decl_start,
                    decl_end=decl_end,
                    is_reduce=binding.schedule.is_reduce(),
                    role=binding.schedule.role(),
                    rebuild_template=rebuild_template,
                    dst_offsets_arg=dst_offsets_arg,
                    axis_var_names=axis_var_names,
                    descriptor_axes=descriptor_axes,
                )
            )
        return tuple(specs)

    def transform(self, source: str) -> str:
        lines = source.splitlines()
        if self.enable_consumer or self.enable_producer or self.remote_descriptors:
            lines = _ensure_import(lines, "import triton_dist.language as dl")
        parsed = parse_annotations(lines)

        remote_specs = self._build_remote_specs(lines, parsed)

        persistent_init_lines: Optional[List[str]] = None
        producer_epilogue_lines: Optional[List[str]] = None
        wave_dim_name = "wave_dim"
        cur_wave_sizes_arg = "cur_wave_sizes"
        wave_offsets_arg = "wave_offsets"

        dispatch_result: Optional[_DispatchGenerationResult]
        if parsed.persistent:
            axis_info = _build_persistent_axis_info(parsed)
            dispatch_result = _generate_persistent_dispatch_block(
                lines,
                parsed,
                axis_info,
                signal_roles=self.signal_roles,
                consumer_role=self.consumer_signal_role,
                consumer_descriptors=self.consumer_descriptors,
                wave_dim_name=wave_dim_name,
                cur_wave_sizes_arg=cur_wave_sizes_arg,
                wave_offsets_arg=wave_offsets_arg,
                remote_specs=remote_specs,
            )
            persistent_init_lines = _generate_persistent_init_block(
                parsed,
                axis_info,
                signal_roles=self.signal_roles,
                wave_dim_name=wave_dim_name,
                cur_wave_sizes_arg=cur_wave_sizes_arg,
                wave_offsets_arg=wave_offsets_arg,
                remote_specs=remote_specs,
            )
            extra_args = self._persistent_extra_args()
        else:
            dispatch_result = _generate_dispatch_block(
                lines,
                parsed,
                signal_roles=self.signal_roles,
                consumer_role=self.consumer_signal_role,
                consumer_descriptors=self.consumer_descriptors,
                enable_producer=self.enable_producer,
                remote_specs=remote_specs,
            )
            extra_args = self._nonpersistent_extra_args()
        if dispatch_result is None:
            raise AnnotationError("Failed to generate dispatch block")
        dispatch_lines = dispatch_result.lines

        if self.enable_producer:
            if parsed.producer_epilogue is None:
                raise AnnotationError(
                    "Producer kernels must mark epilogue with @sy.producer_epilogue"
                )
            if self.producer_signal_role is None or self.producer_counter_arg is None:
                raise AnnotationError("Producer configuration is incomplete")
            signal_var = dispatch_result.signal_vars.get("producer")
            if signal_var is None:
                raise AnnotationError(
                    "Producer dispatch did not expose a signal offset variable"
                )
            tile_count_var = dispatch_result.tile_count_var
            if tile_count_var is None:
                raise AnnotationError("Producer kernels require wave tile counts")
            producer_epilogue_lines = _generate_producer_epilogue_block(
                parsed.producer_epilogue.indent,
                signal_ptr=self.producer_signal_role.ptr_arg,
                signal_offset_var=signal_var,
                counter_ptr=self.producer_counter_arg,
                tile_count_var=tile_count_var,
                wave_index_var=dispatch_result.wave_index_var,
            )
        def_start = _find_def_start(lines, parsed.dispatch.start)
        signature_lines, def_end = _augment_signature_lines(lines, def_start, extra_args)

        remote_decl_index: Dict[int, _RemoteSpec] = {
            spec.decl_start: spec for spec in remote_specs
        }

        # locate the line with the kernel definition we need to patch
        new_lines: List[str] = []
        idx = 0
        while idx < len(lines):
            if idx == def_start:
                new_lines.extend(signature_lines)
                idx = def_end + 1
                continue
            if (
                parsed.persistent
                and parsed.persistent_init is not None
                and parsed.persistent_init.start <= idx <= parsed.persistent_init.end
            ):
                if idx == parsed.persistent_init.start and persistent_init_lines is not None:
                    new_lines.extend(persistent_init_lines)
                idx = parsed.persistent_init.end + 1  # type: ignore[union-attr]
                continue
            if parsed.dispatch.start <= idx <= parsed.dispatch.end:
                if idx == parsed.dispatch.start:
                    new_lines.extend(dispatch_lines)
                idx = parsed.dispatch.end + 1
                continue
            if (
                parsed.producer_epilogue is not None
                and parsed.producer_epilogue.start <= idx <= parsed.producer_epilogue.end
            ):
                if (
                    producer_epilogue_lines is not None
                    and idx == parsed.producer_epilogue.start
                ):
                    new_lines.extend(producer_epilogue_lines)
                idx = parsed.producer_epilogue.end + 1
                continue
            if idx in remote_decl_index:
                spec = remote_decl_index[idx]
                indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
                new_lines.append(
                    f"{indent}{spec.cur_target_rank_var} = tl.load({spec.target_rank_arg})"
                )
                new_lines.append(
                    f"{indent}{spec.remote_ptr_var} = dl.symm_at({spec.base_ptr_arg}, {spec.cur_target_rank_var})"
                )
                new_lines.extend(_reindent(spec.rebuild_template, indent))
                idx = spec.decl_end + 1
                continue
            new_lines.append(lines[idx])
            idx += 1

        if parsed.persistent:
            _rewrite_kernel_num_tiles(new_lines, parsed.axis_counts)
        new_lines = _transform_host_entries(new_lines, parsed.axis_counts)

        for spec in remote_specs:
            if spec.is_reduce and spec.role == "producer":
                _rewrite_store_to_atomic_add(new_lines, spec.descriptor)

        # Adjust each remote descriptor's call-site offsets so the access lands
        # at the peer-side dst position rather than the source-side global one.
        # Mapping is per descriptor position: positions that don't correspond
        # to any annotation axis (e.g. the K dim of a 2-D matmul descriptor)
        # stay untouched.
        for spec in remote_specs:
            adjust_exprs: List[Optional[str]] = []
            for axis_name in spec.descriptor_axes:
                if axis_name is None:
                    adjust_exprs.append(None)
                else:
                    adjust_exprs.append(spec.axis_var_names[axis_name][1])
            _rewrite_descriptor_call_offsets(
                new_lines, spec.descriptor, adjust_exprs
            )

        result = "\n".join(new_lines)
        if source.endswith("\n"):
            result += "\n"
        return result
