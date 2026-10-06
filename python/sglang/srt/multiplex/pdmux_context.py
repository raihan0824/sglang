from dataclasses import dataclass, field
from typing import List

import torch
import yaml
from torch.cuda.streams import ExternalStream

STREAM_GROUPS = []
SM_COUNTS = []
SM_GROUP_NUM = 8  # Default number of SM groups
CURRENT_STREAM_IDX = 0
CURRENT_STREAM_GROUP = None
GREEN_CONTEXTS = []  # keeps the green contexts behind the streams alive


@dataclass
class PDMuxConfig:
    sm_group_num: int = 8
    manual_divisions: List[List[int]] = field(
        default_factory=list
    )  # [prefill_sm, decode_sm, decode_bs_threshold]
    split_forward_token_budget: int = 65536
    decode_bs_divisor: int = 36


def load_pdmux_config(config_path: str) -> PDMuxConfig:
    """Load pdmux configuration from YAML file into a dataclass."""
    if not config_path:
        return PDMuxConfig()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if "sm_group_num" not in raw:
        raise ValueError("Missing required field: sm_group_num")

    if raw["sm_group_num"] < 3:
        raise ValueError("sm_group_num must be >= 3")

    manual_divisions = raw.get("manual_divisions", [])

    expected = raw["sm_group_num"] - 2
    if manual_divisions and len(manual_divisions) != expected:
        raise ValueError(
            f"manual_divisions must have {expected} entries, "
            f"but got {len(manual_divisions)}"
        )

    return PDMuxConfig(
        sm_group_num=raw["sm_group_num"],
        manual_divisions=manual_divisions,
        split_forward_token_budget=raw.get("split_forward_token_budget", 65536),
        decode_bs_divisor=raw.get("decode_bs_divisor", 36),
    )


def get_arch_constraints(compute_capability):
    major, minor = compute_capability
    # green context constraints for different architectures
    if major == 6:
        return 1, 1  # min_per_part, multiple
    elif major == 7:
        return 2, 2
    elif major == 8:
        return 4, 2
    elif major == 9 and minor >= 0:
        return 8, 8
    elif major == 10:
        # Blackwell (B200/B300): green-context splits of 48/100, 64/84, 80/68 and 96/52 SMs create and run on B300.
        return 8, 8
    else:
        raise ValueError(f"Unsupported compute capability: {major}.{minor}")


def divide_sm(total_sms, compute_capability, groups):
    """
    :param total_sms: total sm count on a single GPU
    :param compute_capability: (major, minor)
    :return: SM partition group(prefill sm, decode sm)
    """
    min_per_part, multiple = get_arch_constraints(compute_capability)
    possible_values = [
        x
        for x in range(min_per_part, total_sms - min_per_part + 1, multiple)
        if x >= total_sms - x and total_sms - x >= 16
    ]
    if not possible_values:
        raise ValueError(
            f"No valid partitions found for total SMs {total_sms} "
            f"with constraints (min per part: {min_per_part}, multiple: {multiple})"
        )

    if len(possible_values) >= groups:
        step = max(1, len(possible_values) // groups)
        selected_values = possible_values[::step][:groups]
    else:
        selected_values = possible_values

    divisions = []
    for part1 in selected_values:
        part2 = total_sms - part1
        divisions.append((part1, part2))

    divisions.reverse()  # Reverse to have larger prefill SM first

    return divisions


def create_aligned_greenctx_streams(prefill_sm: int, decode_sm: int, gpu_id: int):
    """Build the decode partition from whole co-scheduled SM groups and give the prefill partition the
    remaining groups plus the SMs left over by the split.

    On B300 (148 SMs) the driver forms only fifteen 8-SM co-scheduled groups; the other 28 SMs cannot host
    the cluster launches cuBLAS and CUTLASS use. The plain split (prefill first, decode takes the remainder)
    hands decode those SMs: a bf16 cuBLAS GEMM on the 28-SM remainder of a 120/28 split fails with
    CUBLAS_STATUS_EXECUTION_FAILED, and a decode-sized GEMM on the 68-SM remainder of 80/68 takes 136 us
    against 102 us on 64 SMs of whole groups (85 us on the full GPU).
    """
    from cuda.bindings import driver as cu

    def check(result):
        err, *values = result if isinstance(result, tuple) else (result,)
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Green context setup failed: {err}")
        return values[0] if len(values) == 1 else values

    sm_type = cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
    device = check(cu.cuDeviceGet(gpu_id))
    full = check(cu.cuDeviceGetDevResource(device, sm_type))
    align = max(1, getattr(full.sm, "smCoscheduledAlignment", 0) or 8)
    groups, num_groups, leftover = check(
        cu.cuDevSmResourceSplitByCount(full.sm.smCount // align, full, 0, align)
    )
    num_decode_groups = max(1, min(num_groups - 1, decode_sm // align))
    prefill_part = list(groups[: num_groups - num_decode_groups])
    if leftover.sm.smCount > 0:
        prefill_part.append(leftover)
    decode_part = list(groups[num_groups - num_decode_groups : num_groups])

    streams, sm_counts = [], []
    for part in (prefill_part, decode_part):
        desc = check(cu.cuDevResourceGenerateDesc(part, len(part)))
        green_ctx = check(
            cu.cuGreenCtxCreate(
                desc, device, cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
            )
        )
        GREEN_CONTEXTS.append(green_ctx)
        sm_counts.append(check(cu.cuGreenCtxGetDevResource(green_ctx, sm_type)).sm.smCount)
        stream = check(
            cu.cuGreenCtxStreamCreate(
                green_ctx, cu.CUstream_flags.CU_STREAM_NON_BLOCKING, 0
            )
        )
        streams.append(
            ExternalStream(stream_ptr=int(stream), device=torch.device(f"cuda:{gpu_id}"))
        )
    return tuple(streams), tuple(sm_counts)


def initialize_stream_groups(gpu_id: int, config: PDMuxConfig):
    from sgl_kernel import spatial

    global \
        STREAM_GROUPS, \
        SM_COUNTS, \
        SM_GROUP_NUM, \
        CURRENT_STREAM_IDX, \
        CURRENT_STREAM_GROUP
    # for pd_multiplexing, Init stream_groups
    device = torch.cuda.current_device()
    total_sm_count = spatial.get_sm_available(gpu_id)
    # (prefill_sm_count, decode_sm_count)
    if config.manual_divisions:
        divisions = [
            (prefill_sm, decode_sm)
            for prefill_sm, decode_sm, _ in config.manual_divisions
        ]
    else:
        divisions = divide_sm(
            total_sm_count,
            torch.cuda.get_device_capability(device),
            config.sm_group_num - 2,
        )

    aligned_groups = torch.cuda.get_device_capability(device)[0] >= 10
    SM_COUNTS = []
    SM_COUNTS.append((total_sm_count, 0))  # Normal stream for prefill
    STREAM_GROUPS = []
    STREAM_GROUPS.append(
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id))
    )  # Normal stream for prefill
    for prefill_sm, decode_sm in divisions:
        if aligned_groups:
            streams, sm_counts = create_aligned_greenctx_streams(
                prefill_sm, decode_sm, gpu_id
            )
            STREAM_GROUPS.append(streams)
            SM_COUNTS.append(sm_counts)
        else:
            STREAM_GROUPS.append(
                (spatial.create_greenctx_stream_by_value(prefill_sm, decode_sm, gpu_id))
            )
            SM_COUNTS.append((prefill_sm, decode_sm))
    SM_COUNTS.append((0, total_sm_count))  # Normal stream for decode
    STREAM_GROUPS.append(
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id))
    )  # Normal stream for decode

    CURRENT_STREAM_IDX = 0
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]


def set_current_stream_idx(idx: int):
    global CURRENT_STREAM_IDX, CURRENT_STREAM_GROUP
    if idx < 0 or idx >= len(STREAM_GROUPS):
        raise ValueError(f"Invalid stream index: {idx}")
    CURRENT_STREAM_IDX = idx
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]


def get_stream_groups() -> list[tuple[torch.cuda.Stream, torch.cuda.Stream]]:
    """Get the stream groups."""
    return STREAM_GROUPS


def get_sm_counts() -> list[tuple[int, int]]:
    """Get the SM counts."""
    return SM_COUNTS


def get_current_stream_idx() -> int:
    """Get the current stream index."""
    return CURRENT_STREAM_IDX
