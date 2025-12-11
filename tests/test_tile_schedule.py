import pytest

from syncopate.communication.descriptor import Signal, SignalType
from syncopate.interface import tile_schedule


TileCoord = tile_schedule.TileCoord
TileBlock = tile_schedule.TileBlock
RawSchedule = tile_schedule.RawSchedule
SimplifiedSchedule = tile_schedule.SimplifiedSchedule
Mod1DSchedule = tile_schedule.Mod1DSchedule


@pytest.fixture
def mod1d_raw_schedule():
    block_infos = [
        [TileBlock(tile=TileCoord(offsets=(0, 0), shape=(2, 4)), signal=None)],
        [TileBlock(tile=TileCoord(offsets=(0, 4), shape=(2, 4)), signal=Signal(type=SignalType.RECEIVE_DATA, dst_rank=0, dst_offset=0))],
    ]
    return RawSchedule(num_waves=2, block_infos=block_infos, global_shape=(2, 8))


@pytest.mark.parametrize(
    "offsets, shape, message",
    [
        ((0,), (), "identical length"),
        ((0, -1), (2, 2), "offsets must be non-negative"),
        ((0, 1), (2, 0), "shape must contain strictly positive sizes"),
    ],
)
def test_tile_coord_validation_errors(offsets, shape, message):
    with pytest.raises(ValueError, match=message):
        TileCoord(offsets=offsets, shape=shape)


def test_tile_coord_ndim_and_ranges():
    tile = TileCoord(offsets=(1, 3), shape=(2, 5))
    assert tile.ndim == 2
    assert tile.as_ranges() == ((1, 3), (3, 8))


def test_raw_schedule_accepts_valid_configuration(mod1d_raw_schedule):
    schedule = mod1d_raw_schedule
    assert schedule.num_waves == 2
    assert schedule.block_infos[1][0].tile.offsets == (0, 4)


def test_raw_schedule_requires_positive_num_waves():
    with pytest.raises(ValueError, match="num_waves must be positive"):
        RawSchedule(num_waves=0, block_infos=[], global_shape=(2, 2))


def test_raw_schedule_requires_matching_lengths():
    with pytest.raises(ValueError, match="block_infos length must match num_waves"):
        RawSchedule(num_waves=1, block_infos=[], global_shape=(2, 2))


def test_raw_schedule_requires_non_empty_waves():
    with pytest.raises(ValueError, match="each wave must contain at least one block"):
        RawSchedule(num_waves=1, block_infos=[[]], global_shape=(2, 2))


def test_raw_schedule_enforces_block_dimensions():
    block_infos = [[TileBlock(tile=TileCoord(offsets=(0,), shape=(4,)))]]
    with pytest.raises(
        ValueError, match="all blocks must have the same number of dimensions as global_shape"
    ):
        RawSchedule(num_waves=1, block_infos=block_infos, global_shape=(4, 4))


def test_mod1d_schedule_extracts_axis_information(mod1d_raw_schedule):
    schedule = Mod1DSchedule(mod1d_raw_schedule)
    assert schedule.schedule_axis == 1
    assert schedule.stride == 4
    assert schedule.starting_offset == 0
    assert schedule.global_shape == (2, 8)


def test_mod1d_schedule_rejects_multiple_blocks_per_wave():
    block = TileBlock(tile=TileCoord(offsets=(0, 0), shape=(2, 4)))
    block_infos = [[block, block], [block]]
    raw = RawSchedule(num_waves=2, block_infos=block_infos, global_shape=(2, 4))
    with pytest.raises(ValueError, match="all waves must contain exactly one block"):
        Mod1DSchedule(raw)


def test_mod1d_schedule_requires_uniform_shapes():
    block_infos = [
        [TileBlock(tile=TileCoord(offsets=(0, 0), shape=(2, 4)))],
        [TileBlock(tile=TileCoord(offsets=(0, 4), shape=(2, 8)))],
    ]
    raw = RawSchedule(num_waves=2, block_infos=block_infos, global_shape=(2, 12))
    with pytest.raises(ValueError, match="all blocks must have the same shape"):
        Mod1DSchedule(raw)


def test_mod1d_schedule_requires_single_axis():
    block_infos = [[TileBlock(tile=TileCoord(offsets=(0, 0), shape=(2, 8)))]]
    raw = RawSchedule(num_waves=1, block_infos=block_infos, global_shape=(2, 8))
    with pytest.raises(ValueError, match="schedule must be defined along exactly one axis"):
        Mod1DSchedule(raw)


def test_simplify_success(mod1d_raw_schedule):
    simplifier = SimplifiedSchedule(origin_schedule=mod1d_raw_schedule)
    assert simplifier.simplify() is True
    assert isinstance(simplifier.simplified_schedule, Mod1DSchedule)


def test_simplify_failure_when_no_simple_schedule_matches():
    block = TileBlock(tile=TileCoord(offsets=(0, 0), shape=(2, 4)))
    block_infos = [[block, block]]
    raw = RawSchedule(num_waves=1, block_infos=block_infos, global_shape=(2, 4))
    simplifier = SimplifiedSchedule(origin_schedule=raw)
    assert simplifier.simplify() is False
    assert simplifier.simplified_schedule is None
