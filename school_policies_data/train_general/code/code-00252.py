    # dicts for faster cross-rank validation
    _groups = {}
    _memberships = {}
    _pg_guids = {}
    for global_rank in pg_config:
        for pg_uid in pg_config[global_rank]:
            desc = pg_config[global_rank][pg_uid]["desc"]
            ranks = ast.literal_eval(pg_config[global_rank][pg_uid]["ranks"])
            # With the adoption of the split_group API, we can have multiple PGs with the same pg_guid (PG Name)
            # So we need to add the hash of all its ranks within the PG as well.
            # Also guid must be a string because `_process_group_name` returns a string.
            pg_guid = pg_uid + str(hash(frozenset(ranks)))
            _pg_guids[(pg_uid, global_rank)] = pg_guid
            if isinstance(ranks, str):
                # TODO Bug in FR data format? ranks is '[0, 1,...]'
                ranks = ast.literal_eval(ranks)

            if pg_guid not in _groups:
                groups.append(Group(id=pg_guid, desc=desc, size=len(ranks)))
                for rank in ranks:
                    memberships.append(Membership(group_id=pg_guid, global_rank=rank))
                _groups[pg_guid] = groups[-1]
                _memberships[pg_guid] = set(ranks)
            else:
                # validation across ranks
                if _groups[pg_guid].desc != desc:
                    raise AssertionError(
                        f"mismatch in desc {_groups[pg_guid].desc} vs {desc} for group {pg_guid}"
                    )
                if _memberships[pg_guid] != set(ranks):
                    raise AssertionError(
                        f"mismatch in membership for group {pg_guid}"
                        f" {_memberships[pg_guid]} vs {set(ranks)}"
                    )
    return groups, _groups, memberships, _memberships, _pg_guids


def build_collectives(
    all_entries: dict[int, list[dict[str, Any]]],
    _groups: dict[str, Group],
    _memberships: dict[str, set[Any]],
    _pg_guids: dict[tuple[str, int], str],
    version: str,
    mismatch_cap: int = 10,
) -> tuple[list[Traceback], list[Collective], list[NCCLCall]]:
    """
    groups, memberships are the non-flat dicts that are indexable
    all_entries is a raw dict from the original dumps:

    all_entries: {
        global_rank: [
            {
                record_id: ordered id of the event in the trace buffer
                pg_id: ProcessGroupNCCL::uid_
                    *note: `pg_id` corresponds to nothing in groups table
                process_group: (pg_name, desc)
                    *note: `pg_name`, `desc` corresponds to `pg_id`, `desc` in groups table
                collective_seq_id: ordered id for collective operations and coalesced group operations
                p2p_seq_id: ordered id for point-to-point operations
                op_id: ordered id including individual ops inside coalescing group
                profiling_name: descriptive name of the operation
                'time_created_ns',
                'input_sizes',
                'output_sizes',
                'state',
                'time_discovered_started_ns',
                'time_discovered_completed_ns',
                'retired',
                'frames',
            }
        ]
    }
    """
    tracebacks: list[Traceback] = []

    collectives: list[Collective] = []
    nccl_calls: list[NCCLCall] = []

    # once we find one mismatch, we stop pairing up collectives since the pairing is possibly incorrect
    # instead, just record the remaining ops as NCCLCalls
    mismatch = {_groups[g].id: 0 for g in _groups}

    # For best effort partial analysis.
    dumps_ranks = {int(key) for key in all_entries}
    """
    - it doesn't matter what order I put collectives/ncclops into their table. we can later on re-sort it by start time
    - there could be multiple options for the "first" collective to pair up (rank 0,1 might do a bcast while rank 2,3 do a bcast)
    - within a group, the first collective must be the same on all ranks in the group, then it can be marked as a
    collective and removed
    """
    while all_entries:
        # we greedily match collectives, starting arbitrarily with the trace from the first rank
        # later, if we exhaust the first rank, we continue with the next 'first rank'
        rank_iter = iter(all_entries)
        first_rank = next(rank_iter)
        other_ranks = list(rank_iter)

        if len(all_entries[first_rank]) == 0:
            all_entries.pop(first_rank)
            continue

        # lets match the first collective! we need to know which ranks are involved, and ensure that this same
        # collective is also the first one on those ranks within that group
        entries = all_entries[first_rank]
        current_entry = entries[0]
        desc = current_entry["process_group"][1]
        # For db build and logs printing, we want to use the original pg_name, not the hash one.
        original_pg_name = current_entry["process_group"][0]
        pg_name = _pg_guids[(original_pg_name, first_rank)]
        expected_ranks = set(_memberships[pg_name])
        entry_state = EntryState(current_entry, expected_ranks)
        match_record = MatchStateRecord(
            expected_ranks=expected_ranks,
            other_ranks=other_ranks,
            entry_state=entry_state,
            candidate_ranks={first_rank},
            candidate_idx={},
            found_ranks=set(),
            found_idx={},
            errors=set(),
        )

        major_v, minor_v = get_version_detail(version)
        find_coalesced_group = (
            find_coalesced_group_p2p_only
            if major_v <= 2 and minor_v < 7
            else find_coalesced_group_with_non_p2p
        )
        maybe_coalesced_group = find_coalesced_group(
            pg_name, entries, _pg_guids, first_rank
        )
        if len(maybe_coalesced_group) > 1:
            num_coalesced_entries = len(maybe_coalesced_group)
            # We need a copy of the original expected ranks to avoid modifying it.
            candidate_ranks = copy.deepcopy(expected_ranks)
            done_ranks = set()
            all_coalesced_entries = {}
            while candidate_ranks:
                curr = candidate_ranks.pop()
                done_ranks.add(curr)
                grp = (
                    find_coalesced_group(pg_name, all_entries[curr], _pg_guids, curr)  # type: ignore[index]
                    if curr in all_entries  # type: ignore[comparison-overlap]
                    else []
                )
                all_coalesced_entries[curr] = grp
                for _, entry in grp:
                    op = Op(entry, _memberships, pg_name)
                    peer = None
                    if op.type == "send":
                        if op._src_g != curr:
                            raise AssertionError(
                                f"Send src error: {curr} expected but {op._src_g} is set"
                            )
                        peer = op._dst_g
                    elif op.type == "recv":
                        if op._dst_g != curr:
                            raise AssertionError(
                                f"Recv dst error: {curr} expected but {op._dst_g} is set"
                            )
                        peer = op._src_g
                    if peer and peer not in done_ranks:
                        candidate_ranks.add(peer)

            if major_v <= 2 and minor_v < 7:
                match = match_coalesced_groups_p2p_only(
                    all_coalesced_entries,
                    group_size=_groups[pg_name].size,
                    groups=_groups,
                    memberships=_memberships,
                    _pg_guids=_pg_guids,
                )
            else:
                match = match_coalesced_groups_with_non_p2p(
                    copy.deepcopy(
                        all_coalesced_entries
                    ),  # We want to keep a copy for cleanup.
                    pg_info=(pg_name, desc),
                    memberships=_memberships,
                    _pg_guids=_pg_guids,
                    mismatch=mismatch,
                    dumps_ranks=dumps_ranks,
                    version=version,
                    collectives=collectives,
                    match_record=match_record,
                )

            if match and mismatch[pg_name] == 0:
                # We treat coalesced collectives as a single collective.
                # TODO: we need to surface a merged collective info like input/output sizes to users.
                collectives.append(
                    match_record.entry_state.to_collective(len(collectives))
                )
            else:
                mismatch[pg_name] += 1
            for r in all_coalesced_entries:
                idx_map = {r: i for i, _ in reversed(all_coalesced_entries[r])}  # noqa: B035
                nccl_calls.extend(
                    reversed(
                        match_record.entry_state.to_nccl_call(
                            all_entries,
                            idx_map,
                            len(nccl_calls),
                            collectives[-1].id if match else None,
                        )
                    )
                )
                # This extra cleanup is needed because we need to pop all collectives within a coalesced collective.
                for i, k in idx_map.items():
                    for _ in range(1, num_coalesced_entries):
                        try:
                            all_entries[i].pop(k)
                        except IndexError:
                            # In the case of a missing rank symptom that a rank didn't schedule the coalesced collective,
                            # we should not fail the analysis script here.
                            pass
        else:
            # Iterate through all the ranks and check if there is a mismatch for the current entry.
            check_current_entry_match(
                all_entries,
                _pg_guids,
                (pg_name, desc),
                current_entry,
                _memberships,
                mismatch,
                match_record,
            )

            # Use heuristics to decide what type of errors and error messages we should print.
            error_analysis(
                all_entries,
                match_record,
                dumps_ranks,
                first_rank,
                current_entry,
                mismatch,
                get_version_detail(version),
                pg_name,
            )

            # at this point there are 3 possibilities
            # 1. we found a match on all the ranks that are members of the group
            #  -> we create a Collective and remove the individual entries from their original lists
            if match_record.found_ranks == expected_ranks and mismatch[pg_name] == 0:
                collectives.append(
                    match_record.entry_state.to_collective(len(collectives))
                )
                idx_map = {
                    r: match_record.found_idx[r] if r != first_rank else 0
                    for r in match_record.found_ranks
                }
                nccl_calls.extend(
                    match_record.entry_state.to_nccl_call(
                        all_entries, idx_map, len(nccl_calls), collectives[-1].id
                    )
                )

            # 2. we found a partial match but some ranks are missing
            # 3. we found no match
            #  -> since it's not a complete collective, no entry goes into collectives but we still record a nccl call
            #     TODO should there be a way to mark 'mismatches'?
            else:
                logger.debug("appending a non-matching collective")
                idx_map = {
                    r: match_record.candidate_idx[r] if r != first_rank else 0
                    for r in match_record.candidate_ranks
                }
                collectives.append(
                    match_record.entry_state.to_collective(
                        len(collectives),
                        errors=match_record.errors,
                        idx_map=idx_map,
                        all_entries=all_entries,
                    )
                )
                nccl_calls.extend(
                    match_record.entry_state.to_nccl_call(
                        all_entries, idx_map, len(nccl_calls), None
                    )
                )

        if mismatch[pg_name] > mismatch_cap:
            logger.error(
                "Too many mismatches for process_group %s: %s aborting", pg_name, desc
            )
            break

    return tracebacks, collectives, nccl_calls


def transform_ft(
    details: dict[str, dict[str, Any]], group_world_size: int
) -> dict[str, dict[str, Any]]:
    for dump_key, dump in details.items():
        rank = dump["rank"]
        for key, pg_config in dump["pg_config"].items():
            if pg_config["desc"] == "default_pg":
                ranks = ast.literal_eval(pg_config["ranks"])
                replica_id = rank // group_world_size
                first_rank = replica_id * group_world_size
                new_ranks = [r + first_rank for r in ranks]
                details[dump_key]["pg_config"][key]["ranks"] = f"{new_ranks}"

    return details


def build_db(
    details: dict[str, dict[str, Any]], args: argparse.Namespace, version: str
) -> Database:
    if args.verbose:
        os.environ["FR_TRACE_VERBOSE_OUTPUT"] = "1"
    # temporary state used for building database
    entries = {}
    pg_config = {}
    version_by_ranks = {}
    for dump in details.values():
        rank = dump["rank"]
        entries[rank] = dump["entries"]
        version_by_ranks[rank] = dump["version"]
        pg_config[rank] = dump["pg_config"]

    # Ensure version is consistent across all ranks.
    check_version(version_by_ranks, version)
    entries = align_trace_from_beginning(entries)
    stack_id_trace_map: dict[str, int] = {}
    if args.just_print_entries:
        entries, stack_id_trace_map = add_stack_id_in_entries(entries)

    # flattened database
    groups, _groups, memberships, _memberships, _pg_guids = build_groups_memberships(
        pg_config
    )
    logger.debug("built groups, memberships")

    if args.just_print_entries:
        just_print_entries(
            entries, _groups, _memberships, _pg_guids, args, stack_id_trace_map
        )
        sys.exit(0)

    if not args.allow_incomplete_ranks:
        check_no_missing_dump_files(entries, memberships)

    tracebacks, collectives, nccl_calls = build_collectives(
        entries, _groups, _memberships, _pg_guids, version, args.mismatch_cap
    )
    logger.debug("built collectives, nccl_calls")
    if args.verbose:
        logger.debug("Groups")
        logger.debug(_tabulate(groups, headers=Group._fields))
        logger.debug("Memberships")
        logger.debug(_tabulate(memberships, headers=Membership._fields))
        logger.debug("Collectives")
        logger.debug(_tabulate(collectives, headers=Collective._fields))
        logger.debug("NCCLCalls")
        logger.debug(_tabulate(nccl_calls, headers=NCCLCall._fields))
    db = Database(
        tracebacks=tracebacks,
        collectives=collectives,
        ncclcalls=nccl_calls,
        groups=groups,
        memberships=memberships,
    )
    return db
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable
from typing import Any


__all__ = ["FlightRecorderLogger"]


class FlightRecorderLogger:
    _instance: Any | None = None
    logger: logging.Logger

    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger("Flight Recorder")

    def __new__(cls) -> Any:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = logging.getLogger("Flight Recorder")
            cls._instance.logger.setLevel(logging.INFO)
            formatter = logging.Formatter("%(message)s")
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            cls._instance.logger.addHandler(ch)
        return cls._instance

    def set_log_level(self, level: int) -> None:
        self.logger.setLevel(level)

    @property
    def debug(self) -> Callable[..., None]:
        return self.logger.debug

    @property
    def info(self) -> Callable[..., None]:
        return self.logger.info

    @property
    def warning(self) -> Callable[..., None]:
        return self.logger.warning

    @property
    def error(self) -> Callable[..., None]:
        return self.logger.error

    @property
    def critical(self) -> Callable[..., None]:
        return self.logger.critical
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
import os
import pickle
import re
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from torch.distributed.flight_recorder.components.fr_logger import FlightRecorderLogger


__all__ = [
    "read_dump",
    "read_dir",
]


logger: FlightRecorderLogger = FlightRecorderLogger()


def read_dump(prefix: str, filename: str) -> dict[str, str | int | list[Any]]:
    basename = os.path.basename(filename)

    rank = int(basename[len(prefix) :])
    host_name = f"host_rank{rank}"

    with open(filename, "rb") as infile:
        dump = pickle.load(infile)

    entries = dump["entries"]
    version = dump["version"]
    pg_config = dump["pg_config"]

    return {
        "host_name": host_name,
        "rank": rank,
        "entries": entries,
        "version": version,
        "pg_config": pg_config,
    }


exp = re.compile(r"([\w\-\_]*?)(\d+)$")


def _determine_prefix(files: list[str]) -> str:
    """If the user doesn't specify a prefix, but does pass a dir full of similarly-prefixed files, we should be able to
    infer the common prefix most of the time.  But if we can't confidently infer, just fall back to requiring the user
    to specify it
    """
    possible_prefixes: defaultdict[str, set[int]] = defaultdict(set)
    for f in files:
        m = exp.search(f)
        if m:
            p, r = m.groups()
            possible_prefixes[p].add(int(r))
    if len(possible_prefixes) == 1:
        prefix = next(iter(possible_prefixes))
        logger.debug("Inferred common prefix %s", prefix)
        return prefix
    else:
        raise ValueError(
            "Unable to automatically determine the common prefix for the trace file names. "
            "Please specify --prefix argument manually"
        )


@contextmanager
def _disable_gc() -> Iterator[None]:
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def read_dir(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], str]:
    with _disable_gc():
        prefix = args.prefix
        details = {}
        t0 = time.time()
        version = ""
        filecount = 0
        if not os.path.isdir(args.trace_dir):
            raise AssertionError(f"folder {args.trace_dir} does not exist")
        for root, _, files in os.walk(args.trace_dir):
            if prefix is None:
                prefix = _determine_prefix(files)
            for f in files:
                if (offset := f.find(prefix)) == -1:
                    continue
                details[f] = read_dump(f[:offset] + prefix, os.path.join(root, f))
                filecount += 1
                if not version:
                    version = str(details[f]["version"])
        tb = time.time()
        if len(details) <= 0:
            raise AssertionError(
                f"no files loaded from {args.trace_dir} with prefix {prefix}"
            )
        logger.debug("loaded %s files in %ss", filecount, tb - t0)
        return details, version
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from enum import auto, Enum
from typing import (
    _eval_type,  # pyrefly: ignore [missing-module-attribute]
    Any,
    Generic,
    NamedTuple,
    TypeVar,
)

from torch.distributed.flight_recorder.components.fr_logger import FlightRecorderLogger


__all__ = [
    "Ref",
    "TypeInfo",
    "MatchState",
    "MatchInfo",
    "Group",
    "Membership",
    "Traceback",
    "Collective",
    "NCCLCall",
    "Database",
    "EntryState",
    "Op",
    "MatchStateRecord",
]


T = TypeVar("T", bound=NamedTuple)


class Ref(Generic[T]):
    pass


class TypeInfo(NamedTuple):
    name: str
    fields: list[tuple[str, type]]  # type: ignore[type-arg]

    @classmethod
    def from_type(cls, c: T) -> "TypeInfo":
        if hasattr(c, "__name__"):
            name = c.__name__
        else:
            name = str(c)
        return cls(
            name,
            [(f, _eval_type(c.__annotations__[f], globals(), {})) for f in c._fields],
        )


class MatchState(Enum):
    """
    Enum representing the possible states of matching for collective operations.

    - FULLY_MATCHED: Indicates that all aspects of the collective operations match.
    - COLLECTIVE_TYPE_MISMATCH: The types of the collective operations differ.
    - SIZE_OR_SYNTAX_MISMATCH: There is a mismatch in input/output sizes or violation of collective syntax.
    - COLLECTIVE_STATE_MISMATCH:
        The states of the collective not same, such as one finished while another just started or scheduled.
    - COLLECTIVE_DTYPE_MISMATCH: The data types of the collective input/output differ.
    - UNDECIDED:
        The match status is ambiguous or cannot be determined, e.g., we might need to check all ranks for alltoall_base.
    """

    FULLY_MATCHED = auto()
    COLLECTIVE_TYPE_MISMATCH = auto()
    SIZE_OR_SYNTAX_MISMATCH = auto()
    COLLECTIVE_STATE_MISMATCH = auto()
    COLLECTIVE_DTYPE_MISMATCH = auto()
    UNDECIDED = auto()


class MatchInfo:
    """
    Aside from the match state, we also store some dynamic info for the match such as the culprit rank
    or collective state that caused the mismatch.
    """

    def __init__(self, state: MatchState, culprit: str | None = None) -> None:
        self._state = state
        self.culprit = culprit

    def __str__(self) -> str:
        details = f", {self.culprit}" if getattr(self, "culprit", None) else ""
        return f"Error type: {self._state.name}{details}"

    @property
    def state(self) -> MatchState:
        return self._state


"""
Schema for flat DB

TODO schemas not yet implemented
# threads as recorded at termination of process
Threads
    id: int
    traceback_id: int
    process_id: int

Process:
    id: int # Same as world groups RANK
    pid: int
    hostname: str

NCCLOp:
    # nccl op implementation details (sends/recv)
    id: int
    nccl_call_id: int

"""


class Group(NamedTuple):
    id: str
    desc: str
    size: int


class Membership(NamedTuple):
    group_id: str
    global_rank: int


class Traceback(NamedTuple):
    id: int
    frames: str


class Collective(NamedTuple):
    id: int
    group_id: str
    pass_check: bool
    collective_seq_id: int
    p2p_seq_id: int
    record_id: int
    pg_desc: str
    collective_name: str
    input_sizes: list[list[int]]
    output_sizes: list[list[int]]
    expected_ranks: set[int]
    collective_state: str
    collective_frames: list[dict[str, str]]
    input_numel: int | None = None
    output_numel: int | None = None
    missing_ranks: set[int] | None = None
    mismatch_collectives: dict[int, "Collective"] | None = None
    type_of_mismatch: MatchInfo | None = None


class NCCLCall(NamedTuple):
    id: int
    # pyrefly: ignore [bad-specialization]
    collective_id: Ref[Collective]
    group_id: str
    global_rank: int  # technically Ref[Process] once we have it
    # pyrefly: ignore [bad-specialization]
    traceback_id: Ref[Traceback]
    collective_type: str
    sizes: list[list[int]]


class Database(NamedTuple):
    groups: list[Group]
    memberships: list[Membership]
    tracebacks: list[Traceback]
    collectives: list[Collective]
    ncclcalls: list[NCCLCall]


# TODO: We need to add a schema for the following
types = [
    TypeInfo.from_type(t)  # type: ignore[type-var]
    for t in [Database, NCCLCall, Collective, Traceback, Membership, Group]
    if (
        isinstance(t, type)
        and issubclass(t, tuple)
        and hasattr(t, "_fields")
        and t is not TypeInfo
    )
]

"""
Stacktrace cache
TODO
"""


"""
Collective Matching logic

NOTE: For now, these collectives need to be supported by NCCL,
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html.
"""
COLLECTIVES = {
    "broadcast",
    "_broadcast_oop",
    "reduce",
    "_reduce_oop",
    "all_gather",
    "all_gather_single",
    "all_gather_v",
    "all_reduce",
    "_all_gather_base",
    "all_gather_into_tensor_coalesced",
    "reduce_scatter",
    "reduce_scatter_single",
    "reduce_scatter_v",
    "reduce_scatter_tensor_coalesced",
    "_reduce_scatter_base",
    "gather",
    "scatter",
    "all_to_all",
    "all_to_all_single",
    "all_to_all_v_single",
    "all_reduce_barrier",
    "barrier",
    "split",
    "new_window",
    "allreduce_coalesced",
    "ALLGATHER_coalesced",
    "REDUCE_SCATTER_coalesced",
}

P2P = {
    "send",
    "recv",
}


class EntryState:
    """
    Util class to keep track of the state of an entry and standardize the way we
    log the error info during analysis.
    """

    def __init__(self, entry: dict[str, Any], expected_ranks: set[int]) -> None:
        self.pg_name = entry["process_group"][0]
        self.desc = entry["process_group"][1]
        self.pg_desc = (
            f"{self.pg_name}:{self.desc}" if self.desc != "undefined" else self.pg_name
        )
        self.profiling_name = entry["profiling_name"]
        self.collective_seq_id = entry["collective_seq_id"]
        self.p2p_seq_id = entry["p2p_seq_id"]
        self.record_id = entry["record_id"]
        self.input_sizes = entry["input_sizes"]
        self.output_sizes = entry["output_sizes"]
        self.collective_state = entry["state"]
        self.collective_frames = entry.get("frames", [])
        self.expected_ranks = expected_ranks
        self.missing_ranks: set[int]
        self.input_numel: int
        self.output_numel: int
        self.errors: set[tuple[int, MatchInfo]]

    def log(
        self,
        logger: FlightRecorderLogger,
        logger_msg: str,
        frame_formatter: Any,
        total_numel: tuple[int, int] | None = None,
        errors: set[tuple[int, MatchInfo]] | None = None,
        missing_ranks: set[int] | None = None,
    ) -> None:
        logger.info(
            logger_msg,
            self.collective_seq_id,
        )
        logger.info("internal record id: %s", self.record_id)
        logger.info("group info: %s", self.pg_desc)
        logger.info("collective: %s", self.profiling_name)
        if missing_ranks:
            self.missing_ranks = missing_ranks
            logger.info("missing ranks: %s", missing_ranks)
        if total_numel:
            self.input_numel = total_numel[0]
            self.output_numel = total_numel[1]
            logger.info("total input numel: %d", total_numel[0])
            logger.info("total output numel: %d", total_numel[1])
        logger.info("input sizes: %s", self.input_sizes)
        logger.info("output sizes: %s", self.output_sizes)
        logger.info("world size: %d", len(self.expected_ranks))
        logger.info("expected ranks: %s", self.expected_ranks)
        logger.info("collective state: %s", self.collective_state)
        if errors:
            self.errors = errors
            error_msg = ", ".join(
                f"Culprit rank {error[0]}; {str(error[1])}" for error in errors
            )
            logger.info("error msg: %s", error_msg)
        logger.info(
            "collective stack trace: \n %s", frame_formatter(self.collective_frames)
        )

    def to_collective(
        self,
        id: int,
        errors: set[tuple[int, MatchInfo]] | None = None,
        idx_map: dict[int, int] | None = None,
        all_entries: dict[int, list[dict[str, Any]]] | None = None,
    ) -> Collective:
        if not errors:
            return Collective(
                id=id,
                group_id=self.pg_name,
                record_id=self.record_id,
                pg_desc=self.pg_desc,
                pass_check=True,
                collective_seq_id=self.collective_seq_id,
                p2p_seq_id=self.p2p_seq_id,
                collective_name=self.profiling_name,
                input_sizes=self.input_sizes,
                output_sizes=self.output_sizes,
                expected_ranks=self.expected_ranks,
                collective_state=self.collective_state,
                collective_frames=self.collective_frames,
                missing_ranks=getattr(self, "missing_ranks", None),
            )
        else:
            if idx_map is None:
                raise AssertionError("idx_map is None")
            if all_entries is None:
                raise AssertionError("all_entries is None")
            mismatch_collectives = {}
            for rank, error in errors:
                idx = idx_map[rank]
                entry = all_entries[rank][idx]
                desc = entry["process_group"][1]
                pg_name = entry["process_group"][0]
                mismatch_collectives[rank] = Collective(
                    id=id,
                    group_id=entry["process_group"][0],
                    record_id=entry["record_id"],
                    pg_desc=f"{pg_name}:{desc}" if desc != "undefined" else pg_name,
                    pass_check=False,
                    collective_seq_id=entry["collective_seq_id"],
                    p2p_seq_id=entry["p2p_seq_id"],
                    collective_name=entry["profiling_name"],
                    input_sizes=entry["input_sizes"],
                    output_sizes=entry["output_sizes"],
                    expected_ranks=self.expected_ranks,
                    collective_state=entry["state"],
                    collective_frames=entry.get("frames", []),
                    type_of_mismatch=error,
                )
            return Collective(
                id=id,
                group_id=self.pg_name,
                record_id=self.record_id,
                pg_desc=self.pg_desc,
                pass_check=False,
                collective_seq_id=self.collective_seq_id,
                p2p_seq_id=self.p2p_seq_id,
                collective_name=self.profiling_name,
                input_sizes=self.input_sizes,
                output_sizes=self.output_sizes,
                expected_ranks=self.expected_ranks,
                collective_state=self.collective_state,
                collective_frames=self.collective_frames,
                input_numel=self.input_numel if hasattr(self, "input_numel") else None,
                output_numel=self.output_numel
                if hasattr(self, "output_numel")
                else None,
                missing_ranks=self.missing_ranks
                if hasattr(self, "missing_ranks")
                else None,
                mismatch_collectives=mismatch_collectives,
            )

    def to_nccl_call(
        self,
        all_entries: dict[int, list[dict[str, Any]]],
        idx_map: dict[int, int],
        nccl_call_id: int,
        collective_id: Any,
    ) -> list[NCCLCall]:
        result = []
        for i, k in idx_map.items():
            all_entries[i].pop(k)
            result.append(
                NCCLCall(
                    id=nccl_call_id,
                    collective_id=collective_id,
                    group_id=self.pg_name,  # type: ignore[arg-type]
                    global_rank=i,
                    traceback_id=0,  # type: ignore[arg-type]
                    collective_type=self.profiling_name,
                    sizes=self.input_sizes,
                )
            )
            nccl_call_id += 1
        return result


class Op:
    """Parses relevant info about operation out of 'event' dict

    examples of supported `profiling_name`s:
        nccl:broadcast
        nccl:send 1->2
        nccl:recv 3<-0
    """

    def __init__(
        self, event: dict[Any, Any], memberships: dict[str, set[Any]], pg_name: str
    ):
        self.profiling_name = event["profiling_name"]
        comm_lib_backend, name = self.profiling_name.split(":")
        _SUPPORTED_BACKENDS = {
            "nccl",
            "ncclx",
            "xccl",
            "gloo",
            "rccl",
            "rcclx",
            "mccl",
            "hccl",
        }
        if comm_lib_backend not in _SUPPORTED_BACKENDS:
            raise AssertionError(
                f"name formatting error? {comm_lib_backend} not in {_SUPPORTED_BACKENDS}"
            )
        parts = name.split(" ")
        type = parts[0]
        meta = parts[1] if len(parts) == 2 else None
        self.state = event["state"]
        # Store the hashed pg_name for accessing memberships, and original pg info for display
        self.pg_name = pg_name  # This is the hashed version used for memberships lookup
        self.original_pg_name, self.pg_desc = event["process_group"]
        if type not in COLLECTIVES | P2P | {"coalesced"}:
            raise AssertionError(f"{type} is not a supported operation")
        self.type = type
        if type == "send":
            if not isinstance(meta, str):
                raise AssertionError
            s, d = meta.split("->")
            self._src, self._dst = int(s), int(d)
        elif type == "recv":
            if not isinstance(meta, str):
                raise AssertionError
            d, s = meta.split("<-")
            self._dst, self._src = int(d), int(s)
        else:
            self._src, self._dst = -1, -1
        self._init_global_src_dst(memberships[pg_name])
        self.pg_size = len(memberships[pg_name])
        if type in P2P | COLLECTIVES:
            self.input_sizes = event["input_sizes"]
            self.output_sizes = event["output_sizes"]
        else:
            self.input_sizes, self.output_sizes = None, None
        self.collective_seq_id = event["collective_seq_id"]
        self.stack_id = event.get("stack_id", -1)
        self.p2p_seq_id = event["p2p_seq_id"]
        self.input_dtypes = event["input_dtypes"]
        self.output_dtypes = event["output_dtypes"]
        self.time_created_ns = event["time_created_ns"]
        self.collective_frames = event.get("frames", [])
        self.is_verbose = os.getenv("FR_TRACE_VERBOSE_OUTPUT", "0") == "1"

    def _init_global_src_dst(self, pg_ranks: set[Any]) -> None:
        pg_ranks_sorted = sorted(pg_ranks)
        self._src_g = pg_ranks_sorted[self._src] if self._src is not None else None
        self._dst_g = pg_ranks_sorted[self._dst] if self._dst is not None else None

    @property
    def src(self) -> int:
        if self.type not in P2P:
            raise AssertionError("can't get src of non-p2p op")
        return self._src

    @property
    def dst(self) -> int:
        if self.type not in P2P:
            raise AssertionError("can't get dst of non-p2p op")
        return self._dst

    def __repr__(self) -> str:
        p2p_info = ""
        if self.type in P2P:
            p2p_info = f"s={self._src_g} d={self._dst_g}"
        if self.is_verbose:
            verbose_info = (
                f"timestamp_created={self.time_created_ns}",
                p2p_info,
                f"input_sizes={self.input_sizes}",
                f"output_sizes={self.output_sizes}",
                f"input_dtypes={self.input_dtypes}",
                f"output_dtypes={self.output_dtypes}",
                "collective_seq_id | p2p_seq_id="
                f"{self.p2p_seq_id if self.type in P2P else self.collective_seq_id}",
                f"pg_name={self.pg_name}",
                f"pg_description={self.pg_desc}",
                f"pg_size={self.pg_size}",
                f"stack_id={self.stack_id}",
                f"state={self.state}",
            )
            return f"{self.type}(%s)" % ", ".join(s for s in verbose_info if s)
        return f"{self.type}(%sinput_sizes={self.input_sizes}, state={self.state})" % (
            f"{p2p_info}, " if p2p_info else ""
        )

    def dtype_mismatch(self, other: "Op") -> bool:
        if (
            (
                self.type not in ["scatter", "gather", "broadcast"]
                and set(self.input_dtypes) != set(self.output_dtypes)
                and self.input_sizes[0]
                and self.output_sizes[0]
            )
            or (
                self.type not in ["scatter", "broadcast"]
                and set(self.input_dtypes) != set(other.input_dtypes)
                and self.input_sizes[0]
                and other.input_sizes[0]
            )
            or (
                self.type != "gather"
                and set(self.output_dtypes) != set(other.output_dtypes)
                and self.output_sizes[0]
                and other.output_sizes[0]
            )
        ):
            return True
        return False

    def match(self, other: "Op") -> MatchInfo:
        # TODO: I think this can validly not match,
        # e.g. if one PG was used for p2p ops between only some of the peers?
        # if self.seq_id != other.seq_id:
        # return False

        if self.type == "send":
            # TODO: We need more states for p2p ops.
            return (
                MatchInfo(MatchState.FULLY_MATCHED)
                if (
                    other.type == "recv"
                    and self.src == other.src
                    and self.dst == other.dst
                    and self.input_sizes == other.output_sizes
                )
                else MatchInfo(MatchState.SIZE_OR_SYNTAX_MISMATCH)
            )
        elif self.type == "recv":
            return (
                MatchInfo(MatchState.FULLY_MATCHED)
                if (
                    other.type == "send"
                    and self.src == other.src
                    and self.dst == other.dst
                    and self.output_sizes == other.input_sizes
                )
                else MatchInfo(MatchState.SIZE_OR_SYNTAX_MISMATCH)
            )
        elif self.type in COLLECTIVES:
            if self.type != other.type:
                return MatchInfo(
                    MatchState.COLLECTIVE_TYPE_MISMATCH,
                    f"Expected collective type: '{self.type}' does not match found collective type: '{other.type}'",
                )
            if (
                self.type not in ["all_to_all", "scatter"]
                and self.input_sizes != other.input_sizes
            ):
                return MatchInfo(
                    MatchState.SIZE_OR_SYNTAX_MISMATCH,
                    f"Expected input sizes: '{self.input_sizes}' does not match found input sizes: "
                    f"'{other.input_sizes}'",
                )
            if (
                self.type not in ["all_to_all", "gather"]
                and self.output_sizes != other.output_sizes
            ):
                return MatchInfo(
                    MatchState.SIZE_OR_SYNTAX_MISMATCH,
                    f"Expected output sizes: '{self.output_sizes}' does not match found output sizes: "
                    f"'{other.output_sizes}'",
                )
            if (
                self.type in ["all_reduce", "allreduce_coalesced"]
                and self.input_sizes != other.output_sizes
            ):
                return MatchInfo(
                    MatchState.SIZE_OR_SYNTAX_MISMATCH,
                    f"Expected input sizes: '{self.input_sizes}' does not match found output sizes: '{other.output_sizes}'",
                )
            if (
                self.type
                in [
                    "all_gather",
                    "all_gather_base",
                    "all_gather_into_tensor_coalesced",
                ]
                and math.prod(other.output_sizes[0])
                != math.prod(self.input_sizes[0]) * self.pg_size
            ):
                return MatchInfo(
                    MatchState.SIZE_OR_SYNTAX_MISMATCH,
                    f"Found input numel '{math.prod(other.input_sizes[0])} * pg size {self.pg_size}' "
                    f"does not match output numel '{math.prod(other.output_sizes[0])}'",
                )
            if (
                self.type
                in [
                    "reduce_scatter",
                    "_reduce_scatter_base",
                    "reduce_scatter_tensor_coalesced",
                ]
                and math.prod(other.input_sizes[0])
                != math.prod(self.output_sizes[0]) * self.pg_size
            ):
                return MatchInfo(
                    MatchState.SIZE_OR_SYNTAX_MISMATCH,
                    f"Found input numel '{math.prod(other.input_sizes[0])}' does not match output numel "
                    f"'{math.prod(other.output_sizes[0])} * pg size {self.pg_size}'",
                )
            if self.dtype_mismatch(other):
                return MatchInfo(
                    MatchState.COLLECTIVE_DTYPE_MISMATCH,
                    f"Expected dtypes: '{set(self.input_dtypes)}' does not "
                    f"match found dtype: '{set(self.output_dtypes)}/"
                    f"{set(other.input_dtypes)}/{set(other.output_dtypes)}'",
                )
            if self.state != other.state:
                # MatchState()
                return MatchInfo(
                    MatchState.COLLECTIVE_STATE_MISMATCH,
                    f"Expected state: '{self.state}' does not match found state: '{other.state}'",
                )
            if self.type == "all_to_all":
                return MatchInfo(MatchState.UNDECIDED)
        elif self.type in [
            "coalesced",
            "ALLGATHER_coalesced",
            "REDUCE_SCATTER_coalesced",
        ]:
            return (
                MatchInfo(MatchState.FULLY_MATCHED)
                if (other.type == self.type)
                else MatchInfo(MatchState.SIZE_OR_SYNTAX_MISMATCH)
            )
        return MatchInfo(MatchState.FULLY_MATCHED)


class MatchStateRecord:
    def __init__(
        self,
        expected_ranks: set[int],
        other_ranks: list[int],
        entry_state: EntryState,
        candidate_ranks: set[int],
        candidate_idx: dict[int, int],
        found_ranks: set[int],
        found_idx: dict[int, int],
        errors: set[tuple[int, MatchInfo]],
    ) -> None:
        self.expected_ranks = expected_ranks
        self.other_ranks = other_ranks
        self.entry_state = entry_state
        self.candidate_ranks = candidate_ranks
        self.candidate_idx = candidate_idx
        self.found_ranks = found_ranks
        self.found_idx = found_idx
        self.errors = errors
        self.has_undecided_case = False

    def reset_for_coalesced(
        self, entry_state: EntryState, candidate_ranks: set[int]
    ) -> None:
        self.entry_state = entry_state
        self.candidate_ranks = candidate_ranks
        self.candidate_idx = {}
        self.found_ranks = set()
        self.found_idx = {}
        self.errors = set()
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
from typing import Any

from torch.distributed.flight_recorder.components.fr_logger import FlightRecorderLogger
from torch.distributed.flight_recorder.components.types import (
    Collective,
    EntryState,
    Group,
    MatchInfo,
    MatchState,
    MatchStateRecord,
    Membership,
    Op,
    P2P,
)


__all__ = [
    "add_stack_id_in_entries",
    "align_trace_from_beginning",
    "check_current_entry_match",
    "check_no_missing_dump_files",
    "check_version",
    "error_analysis",
    "find_coalesced_group",
    "find_coalesced_group_with_non_p2p",
    "get_version_detail",
    "just_print_entries",
    "match_coalesced_groups_with_non_p2p",
    "match_coalesced_groups",
    "format_frame",
    "format_frames",
    "match_one_event",
    "check_size_alltoall",
]

logger: FlightRecorderLogger = FlightRecorderLogger()


try:
    from tabulate import tabulate
except ModuleNotFoundError:
    logger.debug("tabulate is not installed. Proceeding without it.")


def format_frame(frame: dict[str, str]) -> str:
    name = frame["name"]
    filename = frame["filename"]
    line = frame["line"]
    return f"{name} at {filename}:{line}"


def format_frames(frames: list[dict[str, str]]) -> str:
    formatted_frames = []
    for frame in frames:
        formatted_frames.append(format_frame(frame))
    return "\n".join(formatted_frames)


def match_one_event(
    event_a: dict[Any, Any],
    event_b: dict[Any, Any],
    memberships: dict[str, set[Any]],
    pg_name: str,
) -> MatchInfo:
    op_a = Op(event_a, memberships, pg_name)
    op_b = Op(event_b, memberships, pg_name)
    return op_a.match(op_b)


def match_coalesced_groups(
    all_rank_events: dict[Any, Any],
    group_size: int,
    groups: dict[str, Group],
    memberships: dict[str, set[Any]],
    _pg_guids: dict[tuple[str, int], str],
) -> bool:
    """
    all_rank_events: {
        rank: [
            (idx, event_dict)
        ]
    }

    Note: it is possible for event dicts in a coalesced group to be asymmetric.
        e.g. the following events lists form a valid coalescing group
             events0 [send:1]
             events1 [recv:0, send:2]
             events2 [recv:1]

    Rule 1: all ops should find a match
    Rule 2: relative ordering of sends and recvs in one event list can be arbitrary
        e.g.
        events1 [recv:0, send:2]  —> okay
        events1 [send:2, recv:0] —> also okay
    Rule 3: sends to the same dest or recvs from the src should be in a consistent order
        e.g.
        rank0 [send:1 (100B), send:1 (1000B)]
        rank1 [recv:0 (1000B), recv:0 (100B)]   —> not okay
    """
    all_ops = {
        rank: [
            Op(e, memberships, _pg_guids[(e["process_group"][0], rank)])
            for i, e in all_rank_events[rank]
        ]
        for rank in all_rank_events
    }

    def visualize_ops(
        match: bool,
        _pg_guids: dict[tuple[str, int], str],
    ) -> None:
        all_ops = {
            rank: [
                Op(e, memberships, _pg_guids[(e["process_group"][0], rank)])
                for i, e in all_rank_events[rank]
            ]
            for rank in all_rank_events
        }

        i = 0
        row = []
        progress = True
        table = []
        while progress:
            progress = False
            for r in all_ops:
                if len(all_ops[r]) > i:
                    rank, event = all_rank_events[r][i]
                    # Check if the pg_guid exists for this rank and process group
                    pg_key = (event["process_group"][0], rank)
                    if pg_key in _pg_guids:
                        row.append(
                            Op(
                                event,
                                memberships,
                                _pg_guids[pg_key],
                            )
                        )
                    else:
                        # Skip this entry if pg_guid mapping doesn't exist
                        row.append(None)  # type: ignore[arg-type]
                    progress = True
                else:
                    row.append(None)  # type: ignore[arg-type]
            table.append(row)
            row = []
            i += 1
        title = "Match" if match else "MISMATCH"
        logger.info("%s \n", title)
        logger.info("%s", tabulate(table))  # type: ignore[operator]

    # TODO can't verify seq_id bc there might have been valid seq deltas between ranks even within a pg.
    for op_list in all_ops.values():
        if not op_list:
            # print("TODO- not sure if its valid for only some ranks in a PG to participate in a coalesced op?")
            return False
        if op_list[-1].type != "coalesced":
            raise AssertionError
        op_list.pop(-1)

    while all_ops:
        first_rank = next(iter(all_ops))
        my_ops = all_ops[first_rank]

        if len(all_ops[first_rank]) == 0:
            all_ops.pop(first_rank)
            continue

        # lets match the first collective! we need to know which ranks are involved, and ensure that this same
        # collective is also the first one on those ranks within that group
        op = my_ops[0]
        match_idx = -1
        if op.type in P2P:
            dst_global_rank = sorted(memberships[op.pg_name])[op.dst]
            peer_ops = all_ops[dst_global_rank]
            for i, other in enumerate(peer_ops):
                if op.match(other).state == MatchState.FULLY_MATCHED:
                    match_idx = i
                    break
                elif op.dst == other.src:
                    # Rule 3
                    break
                else:
                    # Rule 1
                    continue
        else:
            raise NotImplementedError("coalesced collective ops")
        if match_idx >= 0:
            my_ops.pop(0)
            peer_ops.pop(match_idx)
        else:
            visualize_ops(False, _pg_guids)
            return False

    visualize_ops(True, _pg_guids)
    return True


# We enabled the creating FR entry for non-P2P slow path collective ops in v2.7.
def match_coalesced_groups_with_non_p2p(
    all_rank_events: dict[Any, Any],
    pg_info: tuple[str, str],
    memberships: dict[str, set[Any]],
    _pg_guids: dict[tuple[str, int], str],
    mismatch: dict[str, int],
    dumps_ranks: set[int],
    version: str,
    collectives: list[Collective],
    match_record: MatchStateRecord,
) -> bool:
    """
    all_rank_events: {
        rank: [
            (idx, event_dict)
        ]
    }

    Note: it is possible for event dicts in a coalesced group to be asymmetric.
        e.g. the following events lists form a valid coalescing group
             events0 [send:1]
             events1 [recv:0, send:2]
             events2 [recv:1]

    Rule 1: all ops should find a match
    Rule 2: relative ordering of sends and recvs in one event list can be arbitrary
        e.g.
        events1 [recv:0, send:2]  —> okay
        events1 [send:2, recv:0] —> also okay
    Rule 3: sends to the same dest or recvs from the src should be in a consistent order
        e.g.
        rank0 [send:1 (100B), send:1 (1000B)]
        rank1 [recv:0 (1000B), recv:0 (100B)]   —> not okay
    """
    all_ops = {
        rank: [
            Op(e, memberships, _pg_guids[(e["process_group"][0], rank)])
            for _, e in all_rank_events[rank]
        ]
        for rank in all_rank_events
    }
    is_p2p = any(op.type in P2P for ops in all_ops.values() for op in ops)
    pg_name = pg_info[0]

    def visualize_ops(
        match: bool,
        _pg_guids: dict[tuple[str, int], str],
    ) -> None:
        all_ops = {
            rank: [
                Op(e, memberships, _pg_guids[(e["process_group"][0], rank)])
                for _, e in all_rank_events[rank]
            ]
            for rank in all_rank_events
        }

        i = 0
        row = []
        progress = True
        table = []
        while progress:
            progress = False
            for r in all_ops:
                if len(all_ops[r]) > i:
                    rank, event = all_rank_events[r][i]
                    # Check if the pg_guid exists for this rank and process group
                    pg_key = (event["process_group"][0], rank)
                    if pg_key in _pg_guids:
                        row.append(
                            Op(
                                event,
                                memberships,
                                _pg_guids[pg_key],
                            )
                        )
                    else:
                        # Skip this entry if pg_guid mapping doesn't exist
                        row.append(None)  # type: ignore[arg-type]
                    progress = True
                else:
                    row.append(None)  # type: ignore[arg-type]
            table.append(row)
            row = []
            i += 1
        title = "Match" if match else "MISMATCH"
        logger.info("%s \n", title)
        logger.info("%s", tabulate(table))  # type: ignore[operator]

    # TODO Need to verify no seq_id deltas for P2P ops.
    for rank, op_list in all_ops.items():
        if not op_list:
            logger.error("Rank %s has an empty op list.", rank)
            continue
        if op_list[-1].type == "coalesced" and is_p2p:
            op_list.pop(-1)

    while all_ops:
        first_rank = next(iter(all_ops))
        my_ops = all_ops[first_rank]

        if len(all_ops[first_rank]) == 0:
            all_ops.pop(first_rank)
            continue

        # lets match the first collective! we need to know which ranks are involved, and ensure that this same
        # collective is also the first one on those ranks within that group
        op = my_ops[0]
        match_idx = -1
        if is_p2p:
            dst_global_rank = sorted(memberships[op.pg_name])[op.dst]
            peer_ops = all_ops[dst_global_rank]
            for i, other in enumerate(peer_ops):
                if op.match(other).state == MatchState.FULLY_MATCHED:
                    match_idx = i
                    break
                elif op.dst == other.src:
                    # Rule 3
                    break
                else:
                    # Rule 1
                    continue
            if match_idx >= 0:
                my_ops.pop(0)
                peer_ops.pop(match_idx)
            else:
                visualize_ops(False, _pg_guids)
                return False
        else:
            all_coalesced_entries = {
                rank: [e for _, e in all_rank_events[rank]] for rank in all_rank_events
            }
            current_entry = all_coalesced_entries[first_rank][0]
            my_ops.pop(0)

            match_record.reset_for_coalesced(
                EntryState(current_entry, match_record.expected_ranks),
                {first_rank},
            )

            # Iterate through all the ranks and check if there is a mismatch for the current entry.
            check_current_entry_match(
                all_coalesced_entries,
                _pg_guids,
                pg_info,
                current_entry,
                memberships,
                mismatch,
                match_record,
            )

            # Use heuristics to decide what type of errors and error messages we should print.
            error_analysis(
                all_coalesced_entries,
                match_record,
                dumps_ranks,
                first_rank,
                current_entry,
                mismatch,
                get_version_detail(version),
                pg_info[0],
            )

            # TODO: For now, we only check the correctness of individual collective within a coalesced one in
            # this script. We need to merge  (e.g, input/output sizes) together
            # for downstream consumer.

            # at this point there are 3 possibilities
            # 1. we found a match on all the ranks that are members of the group
            #  -> we create a Collective and remove the individual entries from their original lists
            if (
                match_record.found_ranks == match_record.expected_ranks
                and mismatch[pg_name] == 0
            ):
                # Just pop out this collective.
                idx_map = {
                    r: match_record.found_idx[r] if r != first_rank else 0
                    for r in match_record.found_ranks
                }
                for i, k in idx_map.items():
                    all_rank_events[i].pop(k)
                for r in match_record.found_ranks:
                    if r != first_rank:
                        all_ops[r].pop(0)

            # 2. we found a partial match but some ranks are missing
            # 3. we found no match
            #  -> since its not a complete collective, no entry goes into collectives but we still record a nccl call
            else:
                logger.debug("Non-matching collective inside coalesced group")
                idx_map = {
                    r: match_record.candidate_idx[r] if r != first_rank else 0
                    for r in match_record.candidate_ranks
                }
                collectives.append(
                    match_record.entry_state.to_collective(
                        len(collectives),
                        errors=match_record.errors,
                        idx_map=idx_map,
                        all_entries=all_coalesced_entries,
                    )
                )
                return False

    if is_p2p:
        visualize_ops(True, _pg_guids)
    return True


def check_size_alltoall(alltoall_cases: list[dict[str, Any]]) -> tuple[bool, int, int]:
    input_numel = 0
    output_numel = 0
    for e in alltoall_cases:
        input_numel += math.prod(e["input_sizes"][0])
        output_numel += math.prod(e["output_sizes"][0])
    return input_numel != output_numel, input_numel, output_numel


def check_current_entry_match(
    all_entries: dict[int, list[dict[str, Any]]],
    _pg_guids: dict[tuple[str, int], str],
    pg_info: tuple[str, str],
    current_entry: dict[str, Any],
    _memberships: dict[str, set[Any]],
    mismatch: dict[str, int],
    match_record: MatchStateRecord,
) -> None:
    pg_name, desc = pg_info[0], pg_info[1]
    for o in match_record.expected_ranks.intersection(set(match_record.other_ranks)):
        for i, e in enumerate(all_entries[o]):  # type: ignore[index]
            # step over ops from other PGs
            # only check match state when seq_id matches
            if (
                _pg_guids[(e["process_group"][0], o)] == pg_name
                and e["process_group"][1] == desc
                and e["collective_seq_id"] == match_record.entry_state.collective_seq_id
            ):
                match_info = match_one_event(current_entry, e, _memberships, pg_name)
                if (
                    match_info.state in [MatchState.FULLY_MATCHED, MatchState.UNDECIDED]
                    and mismatch[pg_name] == 0
                ):
                    match_record.found_ranks.add(o)
                    match_record.found_idx[o] = i
                    match_record.has_undecided_case = (
                        match_info.state == MatchState.UNDECIDED
                    )
                else:
                    match_record.candidate_ranks.add(o)
                    match_record.candidate_idx[o] = i
                    if match_info.state not in [
                        MatchState.FULLY_MATCHED,
                        MatchState.UNDECIDED,
                    ]:
                        # Here we assume the current rank is not the source of the error.
                        # But it's possible that the current rank is the culprit, then users will
                        # see lots of normal ranks reported as culprit.
                        # TODO: we need to figure out a better way to handle the case mentioned above.
                        match_record.errors.add((o, match_info))
                break


def error_analysis(
    all_entries: dict[int, list[dict[str, Any]]],
    match_record: MatchStateRecord,
    dumps_ranks: set[int],
    first_rank: int,
    current_entry: dict[str, Any],
    mismatch: dict[str, int],
    version: tuple[int, int],
    pg_name: str,
) -> None:
    major_v, minor_v = version[0], version[1]
    # case one: not every rank join the collective or in the flight recorder.
    if (
        match_record.candidate_ranks | match_record.found_ranks
    ) != match_record.expected_ranks and match_record.expected_ranks - (
        match_record.candidate_ranks | match_record.found_ranks
    ) <= dumps_ranks:
        mismatch[pg_name] += 1
        logger_msg = "Not all ranks joining collective, sequence number: %s"
        missing_ranks = match_record.expected_ranks - (
            match_record.candidate_ranks | match_record.found_ranks
        )
        match_record.entry_state.log(
            logger, logger_msg, format_frames, missing_ranks=missing_ranks
        )
        match_record.candidate_ranks.update(match_record.found_ranks)
        match_record.candidate_idx.update(match_record.found_idx)
        match_record.found_idx.clear()
        match_record.found_ranks.clear()
    # We didn't see any mismatch and all expected ranks are in the dump.
    elif len(
        match_record.candidate_ranks
    ) == 1 and match_record.expected_ranks.issubset(dumps_ranks):
        # case two: alltoall or alltoall_base case.
        if match_record.has_undecided_case:
            alltoall_cases = [current_entry] + [
                all_entries[o][match_record.found_idx[o]]
                for o in match_record.found_ranks
            ]
            fail_check, total_input_numel, total_output_numel = check_size_alltoall(
                alltoall_cases
            )
            if major_v <= 2 and minor_v <= 3:
                # We don't log the input/output sizes for alltoall before v2.4,
                # so we don't consider the size mismatch as an error for now.
                fail_check = False
            if fail_check:
                # When we see errors in all_to_all, it's hard to tell which rank is the source of the error.
                mismatch[pg_name] += 1
                logger_msg = (
                    "Input/output mismatch in the collective sequence number: %s"
                )
                match_record.entry_state.log(
                    logger,
                    logger_msg,
                    format_frames,
                    total_numel=(total_input_numel, total_output_numel),
                )
                match_record.candidate_ranks.update(match_record.found_ranks)
                match_record.candidate_idx.update(match_record.found_idx)
                match_record.found_idx.clear()
                match_record.found_ranks.clear()
                match_record.errors.add(
                    (first_rank, MatchInfo(MatchState.SIZE_OR_SYNTAX_MISMATCH))
                )
            else:
                match_record.found_ranks.update(match_record.candidate_ranks)
                match_record.found_idx.update(match_record.candidate_idx)
                match_record.candidate_idx.clear()
                match_record.candidate_ranks.clear()
        # case three: all joined and everything matches on all ranks.
        else:
            match_record.found_ranks.update(match_record.candidate_ranks)
            match_record.found_idx.update(match_record.candidate_idx)
            match_record.candidate_idx.clear()
            match_record.candidate_ranks.clear()
    # case four: mismatch cases due to not same type, size mismatch or state mismatch.
    elif len(match_record.errors) > 0:
        mismatch[pg_name] += 1
        logger_msg = "Collective sequence number: %s has errors"
        match_record.entry_state.log(
            logger, logger_msg, format_frames, errors=match_record.errors
        )
        match_record.candidate_ranks.update(match_record.found_ranks)
        match_record.candidate_idx.update(match_record.found_idx)
        match_record.found_idx.clear()
        match_record.found_ranks.clear()
    # partial analysis case when we cannot decide what's wrong with this collective entry.
    else:
        match_record.candidate_ranks.update(match_record.found_ranks)
        match_record.candidate_idx.update(match_record.found_idx)
        match_record.found_idx.clear()
        match_record.found_ranks.clear()
        # if any element in expected_ranks not in dumps_ranks.
        if match_record.expected_ranks - dumps_ranks:
            mismatch[pg_name] += 1
            logger.info(
                "We cannot decide what's wrong with this collective entry "
                "because we missed FR dumps from ranks (%s) so we don't have enough "
                "information. If you want to debug further use -j to dump all raw trace",
                str(match_record.expected_ranks - dumps_ranks),
            )
        else:
            logger.info(
                "No errors found for this collective entry, There could be some "
                "other reasons why we see collective timeout."
            )


def find_coalesced_group(
    pg_name: str,
    entries: list[dict[str, Any]],
    _pg_guids: dict[tuple[str, int], str],
    rank: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Given a list of entries, if the collective_seq_id of the first entry matches that of subsequent ones,
    build and return a list of entries terminating in a 'coalesced' op entry all sharing a collective_seq_id
    """
    found = []
    collective_seq_id = None
    for i, e in enumerate(entries):
        if _pg_guids[(e["process_group"][0], rank)] != pg_name:
            continue
        elif collective_seq_id is None:
            collective_seq_id = (
                e["p2p_seq_id"] if e["is_p2p"] else e["collective_seq_id"]
            )
            found.append((i, e))
        elif not e["is_p2p"] and e["collective_seq_id"] == collective_seq_id:
            found.append((i, e))
        elif e["is_p2p"] and e["p2p_seq_id"] == collective_seq_id:
            found.append((i, e))
        else:
            break

    if len(found) > 1:
        if found[-1][1]["profiling_name"] != "nccl:coalesced":
            raise AssertionError
        return found
    return []


# We enabled the creating FR entry for non-P2P slow path collective ops in v2.7.
def find_coalesced_group_with_non_p2p(
    pg_name: str,
    entries: list[dict[str, Any]],
    _pg_guids: dict[tuple[str, int], str],
    rank: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Given a list of entries, if the collective_seq_id of the first entry matches that of subsequent ones,
    build and return a list of entries terminating in a 'coalesced' op entry all sharing a collective_seq_id
    """
    found = []
    collective_seq_id = None
    for i, e in enumerate(entries):
        if _pg_guids[(e["process_group"][0], rank)] != pg_name:
            continue
        elif collective_seq_id is None:
            collective_seq_id = (
                e["p2p_seq_id"] if e["is_p2p"] else e["collective_seq_id"]
            )
            found.append((i, e))
        elif not e["is_p2p"] and e["collective_seq_id"] == collective_seq_id:
            found.append((i, e))
        elif e["is_p2p"] and e["p2p_seq_id"] == collective_seq_id:
            found.append((i, e))
        else:
            break

    if len(found) > 1:
        name = found[-1][1]["profiling_name"]
        if name.startswith("nccl:") and not name.endswith("_coalesced"):
            logger.error("Rank %s does not have a coalesced end.", rank)
        return found
    return []


def just_print_entries(
    all_entries: dict[int, list[dict[str, Any]]],
    _groups: dict[str, Group],
    _memberships: dict[str, set[Any]],
    _pg_guids: dict[tuple[str, int], str],
    args: argparse.Namespace,
    stack_id_trace_map: dict[str, int],
) -> None:
    rows = []
    ranks = sorted(all_entries.keys())
    headers = [
        f"Rank {rank}"
        for rank in ranks
        if args.selected_ranks is None or rank in args.selected_ranks
    ]
    progress = True
    while progress:
        progress = False
        row = []
        for rank in ranks:
            if args.selected_ranks is not None and rank not in args.selected_ranks:
                continue
            if len(all_entries[rank]) == 0:
                row.append("")
            else:
                entry = all_entries[rank].pop(0)
                pg_name = _pg_guids[(entry["process_group"][0], rank)]
                if (
                    args.pg_filters is None
                    or entry["process_group"][1] in args.pg_filters
                    or entry["process_group"][0] in args.pg_filters
                ):
                    row.append(str(Op(entry, _memberships, pg_name)))
                else:
                    row.append("")
                progress = True
        if progress:
            rows.append(row)

    logger.info(tabulate(rows, headers=headers))

    if stack_id_trace_map and args.print_stack_trace:
        headers = ["stack_id", "frame_stack"]
        rows = []

        for frame, stack_id in sorted(
            stack_id_trace_map.items(), key=lambda item: item[1]
        ):
            rows.append([str(stack_id), frame])

        logger.info(tabulate(rows, headers=headers))


def check_no_missing_dump_files(
    entries: dict[int, Any], memberships: list[Membership]
) -> None:
    all_ranks = {int(membership.global_rank) for membership in memberships}
    dumps_ranks = {int(key) for key in entries}
    missing = all_ranks - dumps_ranks
    if len(missing) != 0:
        raise AssertionError(f"Missing dump files from ranks {missing}")


def check_version(version_by_ranks: dict[str, str], version: str) -> None:
    for rank, v in version_by_ranks.items():
        if v != version:
            raise AssertionError(
                f"Rank {rank} has different version {v} from the given version {version}"
            )


def get_version_detail(version: str) -> tuple[int, int]:
    # pyrefly: ignore [bad-assignment]
    version = version.split(".")
    if len(version) != 2:
        raise AssertionError(f"Invalid version {version}")
    major, minor = map(int, version)
    return major, minor


def add_stack_id_in_entries(
    entries: dict[int, list[dict[str, Any]]],
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, int]]:
    stack_id = 0
    stack_id_trace_map = {}
    for rank in entries:
        for dump in entries[rank]:
            if dump.get("frames", []):
                frames = str(dump["frames"])
                if frames not in stack_id_trace_map:
                    stack_id_trace_map[frames] = stack_id
                    dump["stack_id"] = stack_id
                    stack_id += 1
                else:
                    dump["stack_id"] = stack_id_trace_map[frames]
            else:
                dump["stack_id"] = -1

    return entries, stack_id_trace_map


def align_trace_from_beginning(
    entries: dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    """
    Align the trace entries by record ID for entries.
    This function takes a dictionary of rank names to lists of trace entries as input.
    Each trace entry is a dictionary containing information about a collective operation,
    including its unique identifier (`record_id` is monotonically increasing as we write into the ring buffer).
    The function finds the largest starting point across all ranks by taking the maximum
    `record_id` value of the first entry in each rank. Finally, it filters out any
    entries with `record_id` values less than the maximum starting point.
    The function returns the updated dictionary of sorted and filtered trace entries.

    Args:
        entries (Dict[str, List[Dict[str, Any]]]): A dictionary of rank names to lists of trace entries.

    Returns:
        entries (Dict[str, List[Dict[str, Any]]]): Entries sorted by record ID and filtered by the maximum starting point.
    """

    maximum_starting_record_id = 0
    for rank in entries:
        # Although this is a ring buffer, we already sort the entries by `record_id` when dumping, we just
        # need to find the largest starting point. For example, if the buffer has the following entries:
        # Rank 0: [0, 1, 2, 3, 4, 5, 6]
        # Rank 1: [1, 2, 3, 4, 5, 6, 7]
        # Rank 2: [2, 3, 4, 5, 6, 7, 8]
        # Rank 3: [0, 1, 2, 3, 4, 5, None]
        # Then we should start from collective 2 not 0 because any collective before,
        # we don't have complete records from all ranks so we need to ignore them.
        # If we don't have any trace from some ranks, ignore them
        # as well.
        if len(entries[rank]) == 0:
            continue
        first_record_id = entries[rank][0]["record_id"]
        maximum_starting_record_id = max(maximum_starting_record_id, first_record_id)

    for rank in entries:
        entries[rank] = [
            entry
            for entry in entries[rank]
            if entry["record_id"] >= maximum_starting_record_id
        ]

    return entries
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from collections.abc import Sequence

from torch.distributed.flight_recorder.components.fr_logger import FlightRecorderLogger


__all__ = ["JobConfig"]


logger: FlightRecorderLogger = FlightRecorderLogger()


class JobConfig:
    """
    A helper class to manage the script configuration.
    """

    def __init__(self: "JobConfig"):
        self.parser = argparse.ArgumentParser(
            description="PyTorch Flight recorder analyzing script."
        )
        self.parser.add_argument(
            "trace_dir",
            nargs="?",
            help="Directory containing one trace file per rank, named with <prefix>_<rank>.",
        )
        self.parser.add_argument(
            "--selected-ranks",
            default=None,
            nargs="+",
            type=int,
            help="List of ranks we want to show traces for.",
        )
        self.parser.add_argument(
            "--allow-incomplete-ranks",
            action="store_true",
            help=(
                "FR trace require all ranks to have dumps for analysis. "
                "This flag allows best-effort partial analysis of results "
                "and printing of collected data."
            ),
        )
        self.parser.add_argument(
            "--pg-filters",
            default=None,
            nargs="+",
            type=str,
            help=(
                "List of filter strings, it could be pg name or pg desc. "
                "If specified, only show traces for the given pg."
            ),
        )
        self.parser.add_argument("-o", "--output", default=None)
        self.parser.add_argument(
            "-p",
            "--prefix",
            help=(
                "Common filename prefix to strip such that rank can be extracted. "
                "If not specified, will attempt to infer a common prefix."
            ),
            default=None,
        )
        self.parser.add_argument("-j", "--just_print_entries", action="store_true")
        self.parser.add_argument("-v", "--verbose", action="store_true")
        self.parser.add_argument("--print_stack_trace", action="store_true")
        self.parser.add_argument(
            "--mismatch_cap",
            type=int,
            default=10,
            help="Maximum number of mismatches we print (from earliest).",
        )
        self.parser.add_argument(
            "--transform-ft",
            action="store_true",
            help="Transform PG config to use global ranks to analyze traces produced by torchft",
        )
        self.parser.add_argument(
            "--group-world-size",
            type=int,
            default=None,
            help="The number of ranks in 1 torchft replica group. Must be specified if --transform-ft is True",
        )

    def parse_args(self: "JobConfig", args: Sequence[str] | None) -> argparse.Namespace:
        # pyrefly: ignore [bad-assignment]
        args = self.parser.parse_args(args)
        # pyrefly: ignore [missing-attribute]
        if args.selected_ranks is not None:
            # pyrefly: ignore [missing-attribute]
            if not args.just_print_entries:
                raise AssertionError(
                    "Not support selecting ranks without printing entries"
                )
        # pyrefly: ignore [missing-attribute]
        if args.pg_filters is not None:
            # pyrefly: ignore [missing-attribute]
            if not args.just_print_entries:
                raise AssertionError(
                    "Not support selecting pg filters without printing entries"
                )
        # pyrefly: ignore [missing-attribute]
        if args.verbose:
            logger.set_log_level(logging.DEBUG)
        # pyrefly: ignore [bad-return]
        return args
<|endoftext|># mypy: allow-untyped-defs
import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, NamedTuple

import torch
import torch.distributed as dist


__all__ = ["JoinHook", "Joinable", "Join"]


class JoinHook:
    r"""
    This defines a join hook, which provides two entry points in the join context manager.

    Entry points : a main hook, which is called repeatedly while there exists a non-joined
    process, and a post-hook, which is called once all processes have joined.

    To implement a join hook for the generic join context manager, define a
    class that inherits from :class:`JoinHook` and override ``main_hook()`` and
    ``post_hook()`` as appropriate.
    """

    def main_hook(self) -> None:
        r"""Call this hook while there exists a non-joined process to shadow collective communications in a training iteration.

        Training iteration i.e., in one forward pass, backward pass, and optimizer step.
        """

    def post_hook(self, is_last_joiner: bool) -> None:
        r"""
        Call hook after all processes have joined.

        It is passed an additional ``bool`` argument ``is_last_joiner``, which indicates if the rank is one of the last to join.

        Arguments:
            is_last_joiner (bool): ``True`` if the rank is one of the last to
                join; ``False`` otherwise.
        """


class Joinable(ABC):
    r"""
    This defines an abstract base class for joinable classes.

    A joinable class
    (inheriting from :class:`Joinable`) should implement :meth:`join_hook`,
    which returns a :class:`JoinHook` instance, in addition to
    :meth:`join_device` and :meth:`join_process_group` that return device and
    process group information, respectively.
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self._join_config = _JoinConfig.construct_disabled_join_config()

    @abstractmethod
    def join_hook(self, **kwargs) -> JoinHook:
        r"""
        Return a :class:`JoinHook` instance for the given :class:`Joinable`.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.
        """
        ...

    @property
    @abstractmethod
    def join_device(self) -> torch.device:
        r"""Return the device from which to perform collective communications needed by the join context manager."""
        ...

    @property
    @abstractmethod
    def join_process_group(self) -> Any:
        r"""Returns the process group for the collective communications needed by the join context manager itself."""
        ...


class _JoinConfig(NamedTuple):
    r"""This includes all fields needed from a :class:`Joinable` instance for the join context manager side."""

    enable: bool
    throw_on_early_termination: bool
    is_first_joinable: bool

    @staticmethod
    def construct_disabled_join_config():
        r"""Return a :class:`_JoinConfig` instance indicating that join-related logic should be disabled.

        e.g. if the caller is not in a join context manager.
        """
        return _JoinConfig(
            enable=False, throw_on_early_termination=False, is_first_joinable=False
        )


class Join:
    r"""
    This class defines the generic join context manager, which allows custom hooks to be called after a process joins.

    These hooks should shadow the
    collective communications of non-joined processes to prevent hanging and
    erroring and to ensure algorithmic correctness. Refer to :class:`JoinHook`
    for details about the hook definition.

    .. warning::
        The context manager requires each participating :class:`Joinable` to
        call the method :meth:`notify_join_context()` before its own per-
        iteration collective communications to ensure correctness.

    .. warning::
        The context manager requires that all ``process_group`` attributes in
        the :class:`JoinHook` objects are the same. If there are multiple
        :class:`JoinHook` objects, then the ``device`` of the first is used.
        The process group and device information is used for checking for non-
        joined processes and for notifying processes to throw an exception if
        ``throw_on_early_termination`` is enabled, both of which using an all-
        reduce.

    Arguments:
        joinables (List[Joinable]): a list of the participating
            :class:`Joinable` s; their hooks are iterated over in the given
            order.

        enable (bool): a flag enabling uneven input detection; setting to
            ``False`` disables the context manager's functionality and should
            only be set when the user knows the inputs will not be uneven
            (default: ``True``).

        throw_on_early_termination (bool): a flag controlling whether to throw an
            exception upon detecting uneven inputs (default: ``False``).

    Example::

        >>> import os
        >>> import torch
        >>> import torch.distributed as dist
        >>> import torch.multiprocessing as mp
        >>> # xdoctest: +SKIP
        >>> import torch.nn.parallel.DistributedDataParallel as DDP
        >>> import torch.distributed.optim.ZeroRedundancyOptimizer as ZeRO
        >>> from torch.distributed.algorithms.join import Join
        >>>
        >>> # On each spawned worker
        >>> def worker(rank):
        >>>     dist.init_process_group("nccl", rank=rank, world_size=2)
        >>>     model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank])
        >>>     optim = ZeRO(model.parameters(), torch.optim.Adam, lr=0.01)
        >>>     # Rank 1 gets one more input than rank 0
        >>>     inputs = [torch.tensor([1.]).to(rank) for _ in range(10 + rank)]
        >>>     with Join([model, optim]):
        >>>         for input in inputs:
        >>>             loss = model(input).sum()
        >>>             loss.backward()
        >>>             optim.step()
        >>>     # All ranks reach here without hanging/erroring
    """

    def __init__(
        self,
        joinables: list[Joinable],
        enable: bool = True,
        throw_on_early_termination: bool = False,
        **kwargs,
    ):
        if len(joinables) == 0:
            raise ValueError("The join context manager requires at least one joinable")
        self._joinables = joinables
        self._join_hooks = [
            joinable.join_hook(**kwargs) for joinable in self._joinables
        ]
        self._enable = enable
        self._throw_on_early_termination = throw_on_early_termination
        self._set_joinable_configs()
        self._extract_dist_info()

    def _set_joinable_configs(self) -> None:
        r"""Set the :class:`_JoinConfig` of each participating :class:`Joinable`."""
        if len(self._joinables) <= 0:
            raise AssertionError
        is_first_joinable = True
        for joinable in self._joinables:
            joinable._join_config = _JoinConfig(
                enable=self._enable,
                throw_on_early_termination=self._throw_on_early_termination,
                is_first_joinable=is_first_joinable,
            )
            is_first_joinable = False

    def _extract_dist_info(self) -> None:
        r"""
        Extract the process group and device information from the joinables.

        If there are multiple joinables, then the context manager uses the
        first specified device.

        Preconditions:
            ``self._joinables`` is not ``None`` and is non-empty.

        Raises:
            ValueError
                If there are multiple conflicting ``process_group`` attributes
                among the ``Joinable`` objects.
        """
        process_group = None
        device = None
        # pyrefly: ignore [bad-assignment]
        for joinable in self._joinables:
            if process_group is None:
                process_group = joinable.join_process_group
            elif process_group != joinable.join_process_group:
                raise ValueError(
                    "Using join context manager with multiple process groups"
                )
            if device is None:
                device = joinable.join_device
        self._process_group = process_group
        self._rank = dist.get_rank(self._process_group)
        self._device = device

    def __enter__(self): ...

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ):
        r"""
        Repeatedly runs the main hooks until all processes join; then, runs the post-hooks.

        Raises:
            RuntimeError
                If ``throw_on_early_termination=True``.
        """
        if not self._enable or type:
            return  # propagate the exception directly if one was raised

        all_procs_joined = False
        is_last_joiner = True

        i = 0
        WARN_THRESHOLD = 1000
        warnings.simplefilter("once")

        while not all_procs_joined:
            if i > WARN_THRESHOLD:
                warnings.warn(
                    "Detected uneven input skew of greater than "
                    f"{WARN_THRESHOLD}. This means that rank "
                    f"{self._rank} has at least {WARN_THRESHOLD} "
                    f"fewer inputs than other currently-active ranks. "
                    "This level of skew could lead to performance "
                    "degradation during training.",
                    stacklevel=2,
                )
            # Shadow the all-reduce in non-joined processes
            num_nonjoined_procs = self._get_num_nonjoined_procs()
            if num_nonjoined_procs == 0:
                all_procs_joined = True
            else:
                if self._throw_on_early_termination:
                    self._notify_procs_to_terminate()

                # Run main hooks
                for join_hook in self._join_hooks:
                    join_hook.main_hook()

                is_last_joiner = False
                i += 1

        # Run post-hooks
        for join_hook in self._join_hooks:
            join_hook.post_hook(is_last_joiner)

    def _get_num_nonjoined_procs(self):
        r"""Return the number of non-joined processes by shadowing an all-reduce in the non-joined processes."""
        num_nonjoined_procs = torch.zeros(1, device=self._device)
        dist.all_reduce(num_nonjoined_procs, group=self._process_group)
        return num_nonjoined_procs.item()

    def _notify_procs_to_terminate(self):
        r"""Schedule an all-reduce to notify non-joined processes to terminate.

        Also raise a ``RuntimeError`` indicating that the current process has exhausted its inputs.
        """
        ones = torch.ones(1, device=self._device)
        dist.all_reduce(ones, group=self._process_group)
        raise RuntimeError(f"Rank {self._rank} exhausted all inputs.")

    @staticmethod
    def notify_join_context(joinable: Joinable):
        r"""
        Notifies the join context manager that the calling process has not yet joined.

        Then, if ``throw_on_early_termination=True``, checks if uneven inputs have been detected
        (i.e. if one process has already joined) and throws an exception if so.

        This method should be called from a :class:`Joinable` object before
        its per-iteration collective communications. For example, this should
        be called at the beginning of the forward pass in
        :class:`DistributedDataParallel`.

        Only the first :class:`Joinable` object passed into the context
        manager performs the collective communications in this method, and
        for the others, this method is vacuous.

        Arguments:
            joinable (Joinable): the :class:`Joinable` object calling this
                method.

        Returns:
            An async work handle for the all-reduce meant to notify the context
            manager that the process has not yet joined if ``joinable`` is the
            first one passed into the context manager; ``None`` otherwise.
        """
        if not hasattr(joinable, "_join_config"):
            raise AssertionError(
                f"Check that the {type(joinable)} constructor calls the "
                "``Joinable`` constructor"
            )

        join_config = joinable._join_config
        # First joinable is responsible for the collective communications
        if not join_config.is_first_joinable or not join_config.enable:
            return None

        device = joinable.join_device
        process_group = joinable.join_process_group

        # Schedule an all-reduce to indicate that the caller has not yet joined
        ones = torch.ones(1, device=device)
        work = dist.all_reduce(ones, group=process_group, async_op=True)

        if join_config.throw_on_early_termination:
            # Check if uneven inputs have been detected
            zeros = torch.zeros(1, device=device)
            dist.all_reduce(zeros, group=process_group)
            should_throw = zeros.item()
            if should_throw:
                raise RuntimeError(
                    "Detected at least one rank that exhausted inputs. "
                    "Throwing across all ranks."
                )
        return work
<|endoftext|># mypy: allow-untyped-defs
import functools

import torch
import torch.distributed as dist


class DefaultState:
    r"""
    Stores state needed to perform the default communication algorithm within a communication hook.

    Args:
        process_group (ProcessGroup): The process group to be used.
    """

    __slots__ = [
        "process_group",
        "world_size",
        "gradient_predivide_factor",
        "gradient_postdivide_factor",
    ]

    def __init__(self, process_group: dist.ProcessGroup):
        if process_group is None:
            raise ValueError(f"Expected to pass in an explicit ProcessGroup to {self}.")
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        # Setting two factors `self.gradient_predivide_factor`
        # and `self.gradient_postdivide_factor` to avoid underflow and overflow
        self.gradient_predivide_factor = self._get_gradient_predivide_factor(
            self.world_size
        )
        self.gradient_postdivide_factor = (
            self.world_size / self.gradient_predivide_factor
        )

    @staticmethod
    def _get_gradient_predivide_factor(world_size: int) -> float:
        factor: int = 1
        while world_size % factor == 0 and world_size / factor > factor:
            factor *= 2
        return float(factor)


class LowPrecisionState(DefaultState):
    r"""
    Stores state needed to perform gradient communication in a lower precision within a communication hook.

    Communication hook will cast gradients back to the original
    parameter precision specified by ``parameter_type`` (default: torch.float32).
    Builds on top of the :class:`DefaultState`.

    Args:
        parameter_type (torch.dtype): The precision of model's parameters.
        Required for a hook to cast gradients back to a parameter's precision.
    """

    __slots__ = [
        "parameter_type",
    ]

    def __init__(
        self,
        process_group,
        parameter_type=torch.float32,
    ):
        super().__init__(process_group)
        self.parameter_type = parameter_type


def _decompress(state: LowPrecisionState, grad: torch.Tensor):
    """
    Casts gradients back to full parameter precision so that further computation happens in full precision.
    """
    orig_grad_data = grad.data
    grad.data = grad.data.to(state.parameter_type)
    device_type = ""
    try:
        if grad.device.type == "privateuse1":
            device_type = torch._C._get_privateuse1_backend_name()
        else:
            device_type = grad.device.type
        backend = getattr(torch, device_type)
    except AttributeError as e:
        raise AttributeError(
            f"Device {grad.device}  does not have a \
                corresponding backend registered as 'torch.device_type'."
        ) from e

    # Don't let this memory get reused until after the transfer.
    orig_grad_data.record_stream(backend.current_stream())  # type: ignore[arg-type]


def allreduce_hook(state: DefaultState, grad: torch.Tensor):
    r"""
    Implement the  FSDP communication hook for ``all_reduce`` algorithm and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): State information, configures pre- and post-division factors.
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks.
    """
    # Average grad by pre-division factor. Together pre- and post-division factors
    # lead to an overall averaging by world_size, required for consistency with PyTorch DDP.
    # This is a two-step process to avoid potential underflow and overflow.
    if state.gradient_predivide_factor > 1:
        grad.div_(state.gradient_predivide_factor)
    dist.all_reduce(grad, group=state.process_group)
    # Average grad by post-division factor.
    if state.gradient_postdivide_factor > 1:
        grad.div_(state.gradient_postdivide_factor)


def reduce_scatter_hook(state: DefaultState, grad: torch.Tensor, output: torch.Tensor):
    r"""
    Implement the  FSDP communication hook for ``reduce_scatter`` algorithm.

    For sharded FSDP strategies and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): State information, configures pre- and post-division factors.
        grad (torch.Tensor): An unsharded gradient for the local batch that needs to be
        communicated across ranks.
        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.
    """
    # Average grad by pre-division factor.
    if state.gradient_predivide_factor > 1:
        grad.div_(state.gradient_predivide_factor)
    dist.reduce_scatter_single(output, grad, group=state.process_group)
    # Average grad's shard by post-division factor.
    if state.gradient_postdivide_factor > 1:
        output.div_(state.gradient_postdivide_factor)


def _low_precision_hook(
    prec: torch.dtype,
    state: LowPrecisionState,
    grad: torch.Tensor,
    output: torch.Tensor | None,
):
    if grad.dtype != prec:
        grad.data = grad.data.to(prec)
    if output is not None:
        if output.dtype != prec:
            output.data = output.data.to(prec)
        reduce_scatter_hook(state, grad, output)
        _decompress(state, output)
    else:
        allreduce_hook(state, grad)
        _decompress(state, grad)


def fp16_compress_hook(
    state: LowPrecisionState, grad: torch.Tensor, output: torch.Tensor | None = None
):
    r"""
    Implement FSDP communication hook for a simple gradient compression approach.
    Casts ``grad`` to half-precision floating-point format (``torch.float16``).

    It also averages gradients by ``world_size`` in two steps: first it pre-divides gradients by a
    ``state.gradient_predivide_factor``, and after a communication step (``all_reduce`` or ``reduce_scatter``)
    gradients are averaged by a ``state.gradient_postdivide_factor``.
    Once post-division is done, compressed gradients are casted back to parameters' precision.

    Args:
        state (LowPrecisionState): State information, configures pre- and post-division factors, parameters' precision.
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks in a lower precision.
        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.
    """
    fp16_hook = functools.partial(_low_precision_hook, torch.float16)
    return fp16_hook(state, grad, output)


def bf16_compress_hook(
    state: LowPrecisionState, grad: torch.Tensor, output: torch.Tensor | None = None
):
    r"""
    Implement FSDP communication hook for a simple gradient compression approach .
    Casts ``grad`` to half-precision floating-point format.

    It also averages gradients by ``world_size`` in two steps: first it pre-divides gradients by a
    ``state.gradient_predivide_factor``, and after a communication step (``all_reduce`` or ``reduce_scatter``)
    gradients are averaged by a ``state.gradient_postdivide_factor``.
    Once post-division is done, compressed gradients are casted back to parameters' precision.

    Args:
        state (LowPrecisionState): State information, configures pre- and post-division factors, parameters' precision.
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks in a lower precision.
        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.
    """
    bf16_hook = functools.partial(_low_precision_hook, torch.bfloat16)
    return bf16_hook(state, grad, output)
<|endoftext|># mypy: allow-untyped-defs
import logging

import torch
import torch.distributed as dist

from . import default_hooks as default


logger = logging.getLogger(__name__)


class PostLocalSGDState:
    r"""
    Store state for all-reducing gradients globally until given step, then locally after.

    Stores the state for all-reducing gradients globally using ``process_group`` until step ``start_localSGD_iter``,
    and all-reducing gradients locally using ``subgroup`` afterwards.

    If ``process_group`` is ``None``, the global process group will be used.
    If ``subgroup`` is ``None``, the intra-node process group on each machine will be used.

    Additionally, ``post_local_gradient_allreduce`` may be worth tuning,
    because both true and false may give a faster convergence.
    """

    __slots__ = [
        "process_group",
        "subgroup",
        "start_localSGD_iter",
        "post_local_gradient_allreduce",
        "iter",
    ]

    def __init__(
        self,
        process_group,
        subgroup,
        start_localSGD_iter,
        post_local_gradient_allreduce=True,
    ):
        """Initialize state object with given parameters and log when localSGD start."""
        logger.info(
            "Local SGD will be started after %s iterations", start_localSGD_iter
        )

        # The group used for all-reducing gradients globally.
        self.process_group = process_group
        # The group used for all-reducing gradients locally.
        self.subgroup = subgroup
        self.start_localSGD_iter = start_localSGD_iter
        # Allreduce gradients locally since iteration `start_localSGD_iter`.
        # This may help with the convergence efficiency at the cost of relatively cheap intra-subgroup communication.
        self.post_local_gradient_allreduce = post_local_gradient_allreduce
        # Iteration/step in the training loop.
        self.iter = 0

    def maybe_increase_iter(self, bucket):
        """Track iterations and trigger log message at start of local SGD."""
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `iter` when bucket 0 is processed.
        if bucket.is_last():
            self.iter += 1

        if self.iter == self.start_localSGD_iter:
            logger.info("Start to apply local SGD after %s iterations.", self.iter)


def post_localSGD_hook(
    state: PostLocalSGDState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Run post-localSGD algorithm.

    This DDP communication hook is used for running post-localSGD algorithm,
    by combining with a model averaging component (e.g.,
    :class:`~torch.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager`)
    that runs after the optimizer step.

    Args:
        state (PostLocalSGDState): State information to run post-localSGD.
            Users mainly need to tune ``start_localSGD_iter`` to determine when to start local SGD.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PostLocalSGDState(process_group=process_group, subgroup=subgroup,
                                  start_localSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, post_localSGD_hook)
        >>> # Also need to establish a model averaging module and run model averaging after ``optimizer.step()``.
        >>> # Please refer to the examples in ``torch.distributed.algorithms.model_averaging.averagers`` module.
    """
    global_group_to_use = (
        state.process_group if state.process_group is not None else dist.group.WORLD
    )

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.buffer()

    # Run allreduce using `global_group_to_use` in the first `start_localSGD_iter` iterations.
    if state.iter < state.start_localSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(global_group_to_use, input_tensor)  # type: ignore[arg-type]

    # If `post_local_gradient_allreduce` is not set,
    # then no gradient synchronization after the first `start_localSGD_iter` iterations.
    if not state.post_local_gradient_allreduce:
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        fut.set_result(input_tensor)
        return fut

    # Run allreduce using `subgroup` after the first `start_localSGD_iter` iterations.
    # Note that by default, a separate subgroup for each node is created which
    # causes an intra-node allreduce to be done at each training step.
    # From this moment, model averaging should run after the optimizer step,
    # to globally allreduce all the parameters.
    if state.subgroup is None:
        state.subgroup, _ = dist.new_subgroups()
    return default._allreduce_fut(state.subgroup, input_tensor)
<|endoftext|># mypy: allow-untyped-defs
import weakref
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import _OverlapStatus
from torch.nn.parallel.distributed import DistributedDataParallel


__all__ = ["hook_with_zero_step", "hook_with_zero_step_interleaved"]

# Functional optimizers require passing a list of gradients to their `step()`
# method, and ZeRO requires a functional optimizer to overlap with DDP
# Passing a `None` instead of an actual gradient indicates to the optimizer
# to not update the corresponding parameter
_NO_PARAM_UPDATE: None = None


def _perform_local_step(
    bucket: dist.GradBucket,
    zero: ZeroRedundancyOptimizer,
    rank: int,
):
    r"""
    Perform a local optimizer step using the gradients provided by ``bucket``.

    Arguments:
        bucket (dist.GradBucket): the bucket providing the gradients.
        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
            instance to perform the :meth:`_local_step`.
        rank (int): the calling process's rank.

    .. warning::
        This function assumes that appropriate synchronization has taken place
        so that the bucket's gradients can be used.
    """
    overlap_info = zero._overlap_info
    bucket_index = bucket.index()
    if len(zero.optim.param_groups) != 1:
        raise AssertionError(
            "Overlapping DDP with ZeRO only supports a single parameter group"
        )

    # Construct the `gradients` input for the local optimizer step, which
    # expects `None` in a list position to indicate that the corresponding
    # parameter should not be updated
    num_local_optim_params = len(zero.optim.param_groups[0]["params"])
    gradients: list[torch.Tensor | None] = [
        _NO_PARAM_UPDATE for _ in range(num_local_optim_params)
    ]
    if bucket_index not in overlap_info.offsets:
        raise AssertionError(
            f"Bucket index {bucket_index} was not assigned to rank {rank}"
        )
    gradients_offset = overlap_info.offsets[bucket_index]
    bucket_assignment = zero._bucket_assignments_per_rank[rank][bucket_index]
    bucket_offset = bucket_assignment.offset
    length = len(bucket_assignment.parameters)
    bucket_gradients = bucket.gradients()[bucket_offset : bucket_offset + length]
    for i, grad in enumerate(bucket_gradients):
        gradients[gradients_offset + i] = grad

    zero._local_step(gradients)


def _broadcast_bucket(
    bucket_index: int,
    zero: ZeroRedundancyOptimizer,
):
    r"""
    Broadcasts a bucket's parameters.

    Arguments:
        bucket_index (int): the index of the bucket corresponding to the
            parameters to broadcast.
        zero (ZeroRedundancyOptimizer): the calling process's
            :class:`ZeroRedundancyOptimizer` instance.
    """
    overlap_info = zero._overlap_info
    if len(overlap_info.assigned_ranks_per_bucket) <= bucket_index:
        raise AssertionError("`assigned_ranks_per_bucket` is not fully constructed")
    # Sort to ensure the same ordering across ranks
    assigned_ranks = sorted(overlap_info.assigned_ranks_per_bucket[bucket_index])
    if len(assigned_ranks) <= 0:
        raise AssertionError(
            f"Bucket {bucket_index} should be assigned to at least one rank"
        )
    for assigned_rank in assigned_ranks:
        bucket_assignments = zero._bucket_assignments_per_rank[assigned_rank]
        if bucket_index in bucket_assignments:
            send_tensor = bucket_assignments[bucket_index].tensor
            if send_tensor is None:
                raise AssertionError
            overlap_info.broadcast_handles.append(
                dist.broadcast(
                    send_tensor,
                    src=dist.get_global_rank(zero.process_group, assigned_rank),
                    group=zero.process_group,
                    async_op=True,
                )
            )


def _save_ddp_bucket_info(
    bucket: dist.GradBucket,
    zero: ZeroRedundancyOptimizer,
):
    r"""
    Save :class:`DistributedDataParallel` gradient bucket information for :class:`ZeroRedundancyOptimizer` instance ``zero``.

    In particular, this function is meant to be called upon seeing each
    gradient bucket to use when overlapping, meaning it does not save or compute any global
    information.

    Arguments:
        bucket (dist.GradBucket): the current gradient bucket.
        zero (ZeroRedundancyOptimizer): the calling process's
            :class:`ZeroRedundancyOptimizer` instance.
    """
    overlap_info = zero._overlap_info
    bucket_params = bucket.parameters()
    if len(bucket_params) <= 0:
        raise AssertionError("Empty bucket")

    # Save the parameters in the bucket
    overlap_info.params_per_bucket.append(bucket_params)
    if overlap_info.shard_buckets:
        # Additionally save the bucket size for the assignment heuristic to use
        bucket_size = 0
        for param in bucket_params:
            bucket_size += param.numel()
        if overlap_info.total_size is None:
            raise AssertionError
        overlap_info.total_size += bucket_size


def _hook_with_zero_step_setup(
    ddp_ref: weakref.ReferenceType,
    zero: ZeroRedundancyOptimizer,
    bucket: dist.GradBucket,
):
    r"""
    Encapsulate the setup logic for :func:`hook_with_zero_step` and :func:`hook_with_zero_step_interleaved`.

    This means the logic to run in the
    hook before the backward pass and optimizer step can actually be
    overlapped. This is factored out since it is common to both
    :func:`hook_with_zero_step` and :func:`hook_with_zero_step_interleaved`.

    Arguments:
        ddp_ref (weakref.ReferenceType): weak reference to the process's
            :class:`DistributedDataParallel` instance.
        zero (ZeroRedundancyOptimizer): the calling process's
            :class:`ZeroRedundancyOptimizer` instance.
        bucket (dist.GradBucket): the current gradient bucket.
    """
    # Proceed as normal until the DDP buckets have been rebuilt
    if not ddp_ref()._has_rebuilt_buckets:  # type: ignore[union-attr]
        if zero._overlap_info.status != _OverlapStatus.UNINITIALIZED:
            raise AssertionError
        return

    bucket_index = bucket.index()
    overlap_info = zero._overlap_info
    if overlap_info.status == _OverlapStatus.UNINITIALIZED:
        overlap_info.status = _OverlapStatus.DDP_HAS_REBUILT_BUCKETS

    if overlap_info.status == _OverlapStatus.DDP_HAS_REBUILT_BUCKETS:
        if bucket_index == 0 and len(overlap_info.params_per_bucket) > 0:
            # This corresponds to the first bucket of the backward pass
            # immediately after all information has been saved, so we
            # can perform the delayed ZeRO initialization
            zero._init_zero_for_overlap()
        else:
            # Once DDP buckets have been rebuilt but ZeRO has not been
            # properly initialized yet, save the information needed
            _save_ddp_bucket_info(bucket, zero)


def hook_with_zero_step(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future],
    ddp: DistributedDataParallel,
    zero: ZeroRedundancyOptimizer,
    shard_buckets: bool = False,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    r"""
    Modify ``hook`` to overlap :class:`ZeroRedundancyOptimizer` optimizer step with :class:`DistributedDataParallel` backward pass.

    This approach overlaps the optimizer computation and communication with the
    backward communication. In particular, the backward computation proceeds
    contiguously, and the optimizer computation follows, overlapping with
    outstanding backward communication (i.e. all-reduces) and possibly other
    optimizer communication (i.e. broadcasts).
    The optimizer step computation begins after the last gradient bucket computation has finished.

    This approach may be preferred over :meth:`hook_with_zero_step_interleaved`
    if communication is relatively slow compared to computation.

    Arguments:
        hook (Callable[[Any, dist.GradBucket], torch.futures.Future]): the hook
            to modify.
        ddp (DistributedDataParallel): the :class:`DistributedDataParallel`
            instance to use.
        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
            instance to use.
        shard_buckets (bool): if ``True``, then the assignment of each
            :class:`DistributedDataParallel` bucket is partitioned across
            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
            across possibly multiple ranks) to approximate uniformity; if
            ``False``, then each bucket is wholly assigned to a single
            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank).

    Returns:
        The modified hook.

    Raises:
        ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.
        RuntimeError: if using any backend other than NCCL/HCCL since currently
            Gloo may hang.

    .. warning::
        Given the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two or three training iterations do not perform parameter updates in
        the optimizer step, depending on if ``static_graph=False`` or
        ``static_graph=True``, respectively. This is because it needs
        information about the gradient bucketing strategy used by
        :class:`DistributedDataParallel`, which is not finalized until the
        second forward pass if ``static_graph=False`` or until the third
        forward pass if ``static_graph=True``.
    """
    if not zero._overlap_with_ddp:
        raise ValueError(
            "ZeroRedundancyOptimizer must be constructed with "
            "`overlap_with_ddp=True` to use this hook properly"
        )
    ddp_ref = weakref.ref(ddp)

    # NOTE: Gloo may hang with this overlapping approach; see https://github.com/pytorch/pytorch/issues/62300
    pg = dist.get_backend(ddp_ref().process_group)  # type: ignore[union-attr]
    if pg == dist.Backend.GLOO:
        raise RuntimeError(
            "Gloo backend using Overlapping DDP with ZeRO may meet hangs"
        )

    if shard_buckets:
        zero._overlap_info.shard_buckets = True
        zero._overlap_info.total_size = 0

    def hook_with_zero_fn(
        state: Any,
        bucket: dist.GradBucket,
    ) -> torch.futures.Future[torch.Tensor]:
        r"""
        Return :class:`Future` that runs the optimizer step if this corresponds to the last gradient bucket.

        Perform equivalent of :class:`ZeroRedundancyOptimizer` :meth:`step` if ``bucket`` is last gradient bucket.
        The function gives a gradient bucket tensor and
        performs additional computation on the iteration that
        the :class:`DistributedDataParallel` buckets are rebuilt to collect
        information used to implement the modified hook.

        Arguments:
            state (Any): any state for the hook.
            bucket (dist.GradBucket): the :class:`DistributedDataParallel`
                gradient bucket.
        """
        fut = hook(state, bucket)
        _hook_with_zero_step_setup(ddp_ref, zero, bucket)
        if zero._overlap_info.status != _OverlapStatus.INITIALIZED:
            return fut

        overlap_info = zero._overlap_info
        bucket_index = bucket.index()
        rank = zero.global_rank

        if overlap_info.status != _OverlapStatus.INITIALIZED:
            raise AssertionError
        if len(overlap_info.assigned_ranks_per_bucket) <= bucket_index:
            raise AssertionError("`assigned_ranks_per_bucket` is not fully constructed")
        assigned_to_bucket = (
            rank in overlap_info.assigned_ranks_per_bucket[bucket_index]
        )

        # Save the bucket reference and all-reduce future for the final bucket
        if assigned_to_bucket:
            overlap_info.bucket_index_to_bucket[bucket_index] = bucket
            overlap_info.bucket_index_to_future[bucket_index] = fut

        # Check that buckets are indexed incrementally starting from 0 in the
        # order of their autograd hooks firing
        if len(overlap_info.bucket_indices_seen) > 0:
            if overlap_info.bucket_indices_seen[-1] != bucket_index - 1:
                raise AssertionError("Bucket indices are not in incremental order")
        else:
            if bucket_index != 0:
                raise AssertionError("Bucket indices do not start from 0")
        overlap_info.bucket_indices_seen.append(bucket_index)

        # Directly return the future without any optimizer computation if this
        # is not the last bucket
        num_buckets = len(overlap_info.params_per_bucket)
        is_last_bucket = bucket_index == num_buckets - 1
        if not is_last_bucket:
            return fut

        # Perform partial optimizer step on all buckets after the final
        # bucket has been computed
        # NOTE: This should not be chained as a callback to the last bucket's
        # all-reduce future since that would add synchronization that delays
        # all optimizer computation to wait for that last all-reduce
        for bucket_index in range(num_buckets):
            assigned_ranks = overlap_info.assigned_ranks_per_bucket[bucket_index]
            if rank in assigned_ranks:
                # Wait on the bucket's all-reduce future to ensure correct
                # gradients
                if bucket_index not in overlap_info.bucket_index_to_future:
                    raise AssertionError(
                        f"All-reduce future for bucket {bucket_index} not saved "
                        f"on rank {rank}"
                    )
                allreduce_future = overlap_info.bucket_index_to_future[bucket_index]
                allreduce_future.wait()

                # Perform the partial optimizer step
                curr_bucket = overlap_info.bucket_index_to_bucket[bucket_index]
                _perform_local_step(curr_bucket, zero, rank)

            _broadcast_bucket(bucket_index, zero)

        # Ensure that all parameter updates are finished before the
        # next forward pass
        overlap_info.wait_for_broadcasts()
        overlap_info.clear_per_iter_info()

        return fut

    return hook_with_zero_fn


def hook_with_zero_step_interleaved(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future],
    ddp: DistributedDataParallel,
    zero: ZeroRedundancyOptimizer,
    shard_buckets: bool = False,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    r"""
    Modify ``hook`` to overlap :class:`ZeroRedundancyOptimizer` optimizer step with :class:`DistributedDataParallel` backward pass

    This approach overlaps the optimizer computation and communication with the
    backward computation and communication. In particular, once a bucket's
    gradients have been computed, the optimizer computation using those
    gradients is launched (though the actual computation must wait for the
    bucket's all-reduce to complete). This yields an interleaving of all-
    reduces and broadcasts in the communication stream.

    This approach may be preferred over :meth:`hook_with_zero_step` if
    communication is relatively fast compared to computation.

    Arguments:
        hook (Any * dist.GradBucket -> torch.futures.Future): the hook to
            modify.
        ddp (DistributedDataParallel): the :class:`DistributedDataParallel`
            instance to use.
        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
            instance to use.
        shard_buckets (bool): if ``True``, then the assignment of each
            :class:`DistributedDataParallel` bucket is partitioned across
            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
            across possibly multiple ranks) to approximate uniformity; if
            ``False``, then each bucket is wholly assigned to a single
            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank).

    Returns:
        The modified hook.

    Raises:
        ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.
        RuntimeError: if using any backend other than NCCL since currently
            Gloo may hang.

    .. warning::
        Given the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two or three training iterations do not perform parameter updates in
        the optimizer step, depending on if ``static_graph=False`` or
        ``static_graph=True``, respectively. This is because it needs
        information about the gradient bucketing strategy used by
        :class:`DistributedDataParallel`, which is not finalized until the
        second forward pass if ``static_graph=False`` or until the third
        forward pass if ``static_graph=True``.
    """
    if not zero._overlap_with_ddp:
        raise ValueError(
            "ZeroRedundancyOptimizer must be constructed with "
            "`overlap_with_ddp=True` to use this hook properly"
        )
    ddp_ref = weakref.ref(ddp)

    # NOTE: Gloo may hang with this overlapping approach; see https://github.com/pytorch/pytorch/issues/62300
    pg = dist.get_backend(ddp_ref().process_group)  # type: ignore[union-attr]
    if pg == dist.Backend.GLOO:
        raise RuntimeError(
            "Gloo backend using Overlapping DDP with ZeRO may meet hangs"
        )

    if shard_buckets:
        zero._overlap_info.shard_buckets = True
        zero._overlap_info.total_size = 0

    def hook_with_zero_interleaved_fn(
        state,
        bucket: dist.GradBucket,
    ) -> torch.futures.Future[torch.Tensor]:
        r"""
        Return :class:`Future` that gives gradient bucket tensor and performs partial :class:`ZeroRedundancyOptimizer` :meth:`step`.

        This function uses the gradients in the given bucket to perform a partial
        :class:`ZeroRedundancyOptimizer` :meth:`step`

        Arguments:
            state: any state for the hook.
            bucket (dist.GradBucket): the :class:`DistributedDataParallel`
                gradient bucket.
        """
        fut = hook(state, bucket)
        _hook_with_zero_step_setup(ddp_ref, zero, bucket)
        if zero._overlap_info.status != _OverlapStatus.INITIALIZED:
            return fut

        def zero_step(fut: torch.futures.Future) -> torch.Tensor:
            r"""
            Perform partial :class:`ZeroRedundancyOptimizer` :meth:`step` using gradients in the :class:`DistributedDataParallel`.

            Returns:
                A :class:`torch.Tensor` representing the contents of the
                gradient bucket.
            """
            overlap_info = zero._overlap_info
            bucket_index = bucket.index()
            rank = zero.global_rank

            assigned_ranks = overlap_info.assigned_ranks_per_bucket[bucket_index]
            overlap_info.bucket_indices_seen.append(bucket_index)
            if rank in assigned_ranks:
                _perform_local_step(bucket, zero, rank)

            _broadcast_bucket(bucket_index, zero)

            num_buckets = len(overlap_info.params_per_bucket)
            if len(overlap_info.bucket_indices_seen) == num_buckets:
                # Ensure that all parameter updates are finished before the
                # next forward pass
                overlap_info.wait_for_broadcasts()
                overlap_info.clear_per_iter_info()

            return bucket.buffer()

        return fut.then(zero_step)

    return hook_with_zero_interleaved_fn
<|endoftext|>from typing import Any

import torch
from torch.distributed import GradBucket


__all__ = ["noop_hook"]


def noop_hook(_: Any, bucket: GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    Return a future that wraps the input, so it is a no-op that does not incur any communication overheads.

    This hook should **only** be used for headroom analysis of allreduce optimization,
    instead of the normal gradient synchronization.
    For example, if only less than 10% speedup of training time can be observed after this hook is registered,
    it usually implies that allreduce is not a performance bottleneck for this case.
    Such instrumentation can be particularly useful
    if GPU traces cannot be easily retrieved or the trace analysis is complicated by
    some factors such as the overlap between allreduce and computation or the desynchronization across ranks.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(None, noop_hook)
    """
    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(bucket.buffer())

    return fut
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, no_type_check

import torch
import torch.distributed as dist
from torch.autograd import Variable


__all__: list[str] = []

_FUNCTIONAL_OPTIM_STEP_METHOD_NAME = "step_param"


class _OptimizerHookState:
    """
    Holds state for running optimizer in-line after DDP communication hook.

    Currently contains only optimizer class which must have a method `step_param`.
    """

    __slots__ = ["functional_optimizer", "params_to_optimize"]

    def __init__(self, functional_optim, params=None):
        self.functional_optimizer = functional_optim
        self._check_valid_functional_optim()
        self._set_params_to_optimize(params)

    def _set_params_to_optimize(self, params):
        if params is not None:
            self.params_to_optimize = set(params)

    def _check_valid_functional_optim(self):
        if not hasattr(self.functional_optimizer, _FUNCTIONAL_OPTIM_STEP_METHOD_NAME):
            raise ValueError(
                f"Class {type(self.functional_optimizer)} must implement method "
                f"{_FUNCTIONAL_OPTIM_STEP_METHOD_NAME}."
            )


@dataclass
class _OptimInBackwardHookState:
    optim_stream: torch.Stream
    wait_for_optim_stream_enqueued: bool


@no_type_check
def _apply_optim_in_backward_hook(
    gradient_is_bucket_view: bool,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    r"""
    Register hook to apply the optimizer in backward.

    If torch.distributed.optim._apply_optimizer_in_backward is used to overlap
    optimizer with backward pass, DDP will run the below hook to run optimizer
    step for parameters after gradient communication has taken place.
    """
    optim_in_bwd_state = _OptimInBackwardHookState(
        optim_stream=torch.Stream(),
        wait_for_optim_stream_enqueued=False,
    )

    def apply_optim_in_backward_hook(
        hook_state: Any,
        bucket: dist.GradBucket,
        optim_stream_state,
    ) -> torch.futures.Future[torch.Tensor]:
        # Run original hook
        ddp_weakref = hook_state
        ddp_inst = ddp_weakref()
        reducer, process_group = ddp_inst.reducer, ddp_inst.process_group
        fut = reducer._run_allreduce_hook(bucket)
        optimizer_stream = optim_stream_state.optim_stream
        with optimizer_stream:
            fut.wait()
            # Apply gradient division since C++ side only allreduces and does
            # not average. TODO: (rohan-varma) the div factor may be different
            # when running with join hook
            bucket.buffer().div_(process_group.size())
            model_params = bucket.parameters()
            grads = bucket.gradients()
            # TODO (rohan-varma): upcast as needed for DDP mixed precision,
            # once optimizer in backward + DDP mixed precision is supported.
            for p, g in zip(model_params, grads):
                if hasattr(p, "_in_backward_optimizers"):
                    # Note: need to set grad to the bucket's grad, because
                    # running allreduce results in the bucket's grad being
                    # reduced, but not grad field.
                    if not gradient_is_bucket_view:
                        p.grad = g
                    for optim in p._in_backward_optimizers:
                        optim.step()

        # Need to return a Future[Tensor] to obey comm hook API contract.
        ret_fut = torch.futures.Future()
        ret_fut.set_result(bucket.buffer())

        # enqueue a callback to wait for this optimizer stream at the end of
        # backward and set all DDP managed grads to None.
        def wait_for_optim_stream_callback():
            torch.accelerator.current_stream().wait_stream(
                optim_stream_state.optim_stream
            )
            # Set DDP managed grads to None
            for param in ddp_inst._get_data_parallel_params(ddp_inst.module):
                if hasattr(param, "_in_backward_optimizers"):
                    param.grad = None

            # reset for the next backward pass
            optim_stream_state.wait_for_optim_stream_enqueued = False

        if not optim_stream_state.wait_for_optim_stream_enqueued:
            Variable._execution_engine.queue_callback(wait_for_optim_stream_callback)
            # mark that the callback is enqueued
            optim_stream_state.wait_for_optim_stream_enqueued = True

        return ret_fut

    comm_hook = partial(
        apply_optim_in_backward_hook, optim_stream_state=optim_in_bwd_state
    )
    # These are needed for DDP's logging of comm hooks
    comm_hook.__name__ = apply_optim_in_backward_hook.__name__
    comm_hook.__qualname__ = apply_optim_in_backward_hook.__qualname__

    return comm_hook


def _hook_then_optimizer(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]],
    optimizer_state: _OptimizerHookState,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    r"""Run optimizer in a functional fashion after DDP communication hook."""
    has_set_params = (
        hasattr(optimizer_state, "params_to_optimize")
        and optimizer_state.params_to_optimize is not None
    )

    def hook_then_optimizer_wrapper(
        hook_state, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # Run original hook
        fut = hook(hook_state, bucket)

        def optimizer_step(fut):
            gradient_tensors = bucket.gradients()
            model_params = bucket.parameters()
            for grad_tensor, model_param in zip(gradient_tensors, model_params):
                if (
                    not has_set_params
                    or model_param in optimizer_state.params_to_optimize
                ):
                    optimizer_state.functional_optimizer.step_param(
                        model_param,
                        grad_tensor,
                    )
            return bucket.buffer()

        return fut.then(optimizer_step)

    return hook_then_optimizer_wrapper
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch.distributed as dist
from torch import nn


def _quantize_per_tensor_backend(x, scale, zero_point):
    y = torch.round(x / scale) + zero_point
    y = torch.clamp(y, 0, 255).to(torch.uint8)
    return y


def _dequantize_per_tensor_backend(y, scale, zero_point):
    x = scale * (y.to(torch.float32) - zero_point)
    return x


def _quantize_per_channel_backend(x, scale, zero_point):
    y = torch.zeros(x.size(), device=x.device)
    for i in range(x.size()[0]):
        y[i, :] = torch.round(x[i, :] / scale[i]) + zero_point[i]
    y = torch.clamp(y, 0, 255).to(torch.uint8)
    return y


def _dequantize_per_channel_backend(y, scale, zero_point):
    y = y.to(torch.float32).to(y.device)
    x = torch.zeros_like(y, device=y.device)
    for i in range(x.size()[0]):
        x[i, :] = scale[i] * (y[i, :] - zero_point[i])
    return x


def _get_allgather_out_list(all_gather_in_list, world_size):
    out_list = [
        torch.zeros_like(
            all_gather_in_list,
            device=all_gather_in_list.device,
            dtype=all_gather_in_list.dtype,
        )
        for _ in range(world_size)
    ]
    return out_list


def quantization_pertensor_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Apply ``torch.quantize_per_tensor`` logic to DDP using ``allgather`` protocol.

    Workers first allgather the scale and zero point of their own
    ``GradBucket`` prior to the quantization. After all workers have that information,
    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's
    own gradient tensor, and uses ``allgather`` to communicate these across all workers.
    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes and
    aggregates each quantized gradient tensor locally and returns the mean.

    .. warning ::
        This is experimental, and uses ``allgather`` protocol which is considerably slower than
        ``allreduce`` protocol. It works only with flattened grads.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, quantization_pertensor_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    # pyrefly: ignore [missing-attribute]
    world_size = group_to_use.size()

    tensor = bucket.buffer()

    myObserver = torch.ao.quantization.MinMaxObserver().to(tensor.device)
    myObserver(tensor)

    s, z = myObserver.calculate_qparams()
    s_and_z = torch.FloatTensor([s, z]).to(tensor.device)

    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)

    # First, allgather scale and zeros.
    fut = dist.all_gather(
        all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True
    ).get_future()

    def quantize_and_allgather(fut):
        # Store scale and zeros across all workers.
        all_ranks_s_and_z = fut.wait()[0]
        # All workers quantize their own ``GradBucket`` tensors.
        quantized_tensor = _quantize_per_tensor_backend(
            tensor, all_ranks_s_and_z[rank][0], all_ranks_s_and_z[rank][1]
        )
        # Allgather quantized tensors.
        fut = dist.all_gather(
            _get_allgather_out_list(quantized_tensor, world_size),
            quantized_tensor,
            group=group_to_use,
            async_op=True,
        ).get_future()

        return fut.wait()

    def dequantize_and_aggregate(fut):
        all_ranks_quantized_tensor = fut.wait()[0]

        aggregated_dequantized_tensor = torch.zeros_like(
            all_ranks_quantized_tensor[0], device=tensor.device, dtype=torch.float32
        )
        # Using previously allgathered scales and zeros, dequantize gradient tensors
        # locally and then aggregate them.
        for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_tensor_backend(
                quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1]
            )

        return aggregated_dequantized_tensor / world_size

    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)


def quantization_perchannel_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket, bucket_size=512
) -> torch.futures.Future[torch.Tensor]:
    """
    Apply``torch.quantize_per_channel`` logic to DDP using ``allgather`` protocol.

    Compared to per-tensor, the main motivation of per-channel is
    for considerably large tensors such as a tensor that contains 6 million
    elements quantizing per a bucket size of 512 (or 128) elements may significantly
    increase the resolution.

    It first splits ``GradBucket`` tensor into multiple chunks (channels) of ``bucket_size``
    elements. Then, workers allgather the scales and zero points of their own
    ``GradBucket`` prior to the quantization. After all workers have that information,
    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's
    own gradient tensor, and uses ``allgather`` to communicate these across all workers.
    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes, flattens, and
    aggregates each quantized gradient tensor locally and returns the mean.

    .. warning ::
        This is experimental, and uses ``allgather`` protocol which is considerably slower than
        ``allreduce`` protocol. It works only with flattened grads.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, quantization_perchannel_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    # pyrefly: ignore [missing-attribute]
    world_size = group_to_use.size()

    tensor = bucket.buffer()

    tensor_in_channels = (
        nn.functional.pad(
            input=tensor,
            pad=(0, bucket_size - len(tensor) % bucket_size),
            mode="constant",
            value=0,
        )
        .view(-1, bucket_size)
        .to(tensor.device)
    )

    myPerChannelObserver = torch.ao.quantization.PerChannelMinMaxObserver().to(
        tensor.device
    )
    myPerChannelObserver(tensor_in_channels)

    s_ch, z_ch = myPerChannelObserver.calculate_qparams()
    s_and_z = torch.stack((s_ch, z_ch)).to(tensor.device)

    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)
    # First, allgather scale and zeros.
    fut = dist.all_gather(
        all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True
    ).get_future()

    def quantize_and_allgather(fut):
        # Store scale and zeros across all workers.
        all_ranks_s_and_z = fut.wait()[0]
        # All workers quantize their corresponding ``GradBucket`` tensors.
        quantized_tensor = _quantize_per_channel_backend(
            tensor_in_channels,
            all_ranks_s_and_z[rank, 0, :],
            all_ranks_s_and_z[rank, 1, :],
        )
        # Allgather quantized tensors.
        fut = dist.all_gather(
            _get_allgather_out_list(quantized_tensor, world_size),
            quantized_tensor,
            group=group_to_use,
            async_op=True,
        ).get_future()

        return fut.wait()

    def dequantize_and_aggregate(fut):
        all_ranks_quantized_tensor = fut.wait()[0]

        aggregated_dequantized_tensor = torch.zeros_like(
            all_ranks_quantized_tensor[0], device=tensor.device, dtype=torch.float32
        )
        # Using previously allgathered scales and zeros, dequantize gradient tensors
        # locally and then aggregate them.
        for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_channel_backend(
                quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1]
            )

        return (
            torch.flatten(aggregated_dequantized_tensor).to(tensor.device)[
                : tensor.size()[0]
            ]
            / world_size
        )

    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.distributed as dist


__all__ = [
    "allreduce_hook",
    "fp16_compress_hook",
    "bf16_compress_hook",
    "fp16_compress_wrapper",
    "bf16_compress_wrapper",
]


def _allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    """Average the input gradient tensor by allreduce and returns a future."""
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    # pyrefly: ignore [missing-attribute]
    tensor.div_(group_to_use.size())

    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )


def allreduce_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Call ``allreduce`` using ``GradBucket`` tensors.

    Once gradient tensors are aggregated across all workers, its ``then``
    callback takes the mean and returns the result.

    If user registers this DDP communication hook,
    DDP results is expected to be same as the case where no hook was registered.
    Hence, this won't change behavior of DDP and user can use this as a reference
    or modify this hook to log useful information or any other purposes while
    unaffecting DDP behavior.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, allreduce_hook)
    """
    return _allreduce_fut(process_group, bucket.buffer())


def _compress_hook(
    dtype: torch.dtype,
    process_group: dist.ProcessGroup,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    # pyrefly: ignore [missing-attribute]
    world_size = group_to_use.size()

    buffer = (
        cast(tuple[torch.Tensor, ...], bucket)[0]
        if isinstance(bucket, tuple)
        else bucket.buffer()
    )
    compressed_tensor = buffer.to(dtype).div_(world_size)

    def decompress(fut):
        decompressed_tensor = buffer
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        value = fut if isinstance(fut, torch.Tensor) else fut.value()[0]
        decompressed_tensor.copy_(value)
        return decompressed_tensor

    if torch.compiler.is_compiling():
        grad = dist._functional_collectives.all_reduce(
            compressed_tensor,
            "sum",
            # pyrefly: ignore [bad-argument-type]
            group_to_use,
        )
        return decompress(grad)
    else:
        fut = dist.all_reduce(
            compressed_tensor, group=group_to_use, async_op=True
        ).get_future()
        return fut.then(decompress)


def fp16_compress_hook(
    process_group: dist.ProcessGroup,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """
    Compress by casting ``GradBucket`` to ``torch.float16`` divided by process group size.

    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    return _compress_hook(torch.float16, process_group, bucket)


def bf16_compress_hook(
    process_group: dist.ProcessGroup,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """
    Warning: This API is experimental, and it requires NCCL version later than 2.9.6.

    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision
    `Brain floating point format <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_ (``torch.bfloat16``)
    and then divides it by the process group size.
    It allreduces those ``bfloat16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, bf16_compress_hook)
    """
    return _compress_hook(torch.bfloat16, process_group, bucket)


def fp16_compress_wrapper(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]],
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """
    Cast input tensor to ``torch.float16``, cast result of hook back to input dtype.

    This wrapper casts the input gradient tensor of a given DDP communication hook to half-precision
    floating point format (``torch.float16``), and casts the resulting tensor of the given hook back to
    the input data type, such as ``float32``.
    Therefore, ``fp16_compress_hook`` is equivalent to ``fp16_compress_wrapper(allreduce_hook)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, fp16_compress_wrapper(powerSGD_hook))
    """

    def fp16_compress_wrapper_hook(
        hook_state, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # Cast bucket tensor to FP16.
        bucket.set_buffer(bucket.buffer().to(torch.float16))

        fut = hook(hook_state, bucket)

        def decompress(fut):
            decompressed_tensor = bucket.buffer()
            # Decompress in place to reduce the peak memory.
            # See: https://github.com/pytorch/pytorch/issues/45968
            decompressed_tensor.copy_(fut.value())
            return decompressed_tensor

        # Decompress after hook has run.
        return fut.then(decompress)

    return fp16_compress_wrapper_hook


def bf16_compress_wrapper(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]],
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """
    Warning: This API is experimental, and it requires NCCL version later than 2.9.6.

    This wrapper casts the input gradient tensor of a given DDP communication hook to half-precision
    `Brain floating point format <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_  (``torch.bfloat16``),
    and casts the resulting tensor of the given hook back to the input data type, such as ``float32``.

    Therefore, ``bf16_compress_hook`` is equivalent to ``bf16_compress_wrapper(allreduce_hook)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, bf16_compress_wrapper(powerSGD_hook))
    """

    def bf16_compress_wrapper_hook(
        hook_state, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # Cast bucket tensor to BF16.
        bucket.set_buffer(bucket.buffer().to(torch.bfloat16))

        fut = hook(hook_state, bucket)

        def decompress(fut):
            decompressed_tensor = bucket.buffer()
            # Decompress in place to reduce the peak memory.
            # See: https://github.com/pytorch/pytorch/issues/45968
            decompressed_tensor.copy_(fut.value())
            return decompressed_tensor

        # Decompress after hook has run.
        return fut.then(decompress)

    return bf16_compress_wrapper_hook
<|endoftext|># mypy: allow-untyped-defs
import sys
from enum import Enum
from functools import partial


# To suppress FutureWarning from partial since 3.13
if sys.version_info >= (3, 11):
    # member was introduced in Python 3.11
    from enum import member

    def _enum_member(x):
        return member(x)
else:

    def _enum_member(x):
        return x


import torch.distributed as dist

from . import (
    debugging_hooks as debugging,
    default_hooks as default,
    optimizer_overlap_hooks as optimizer_overlap,
    powerSGD_hook as powerSGD,
    quantization_hooks as quantization,
)


__all__ = ["DDPCommHookType", "register_ddp_comm_hook"]


def _ddp_comm_hook_wrapper(comm_hook, model, state):
    model.register_comm_hook(state, comm_hook)


def _powerSGD_comm_hook_wrapper(
    comm_hook,
    model,
    state,
    matrix_approximation_rank,
    start_powerSGD_iter=1_000,
):
    """
    Wrap PowerSGD communication hook.

    To be consistent with the wrappers of other DDP comm hooks, the input state only needs to be a process group,
    which will be wrapped up with other state info.
    """
    powerSGD_state = powerSGD.PowerSGDState(
        process_group=state,
        matrix_approximation_rank=matrix_approximation_rank,
        start_powerSGD_iter=start_powerSGD_iter,
    )
    model.register_comm_hook(powerSGD_state, comm_hook)


class DDPCommHookType(Enum):
    """
    Enumerate ``ddp_comm_hooks`` and ``ddp_comm_hook_wrapper`` communication hook types.

    DDPCommHookType enumerates the hooks of ``torch.distributed.algorithms.ddp_comm_hooks``
    as names and ``ddp_comm_hook_wrapper`` partials with hook specified. As an example,
    you can register allreduce hook by
    ``DDPCommHookType.ALLREDUCE.value(model=model, state=process_group)``.
    """

    ALLREDUCE = _enum_member(
        partial(_ddp_comm_hook_wrapper, comm_hook=default.allreduce_hook)
    )
    FP16_COMPRESS = _enum_member(
        partial(_ddp_comm_hook_wrapper, comm_hook=default.fp16_compress_hook)
    )
    BF16_COMPRESS = _enum_member(
        partial(_ddp_comm_hook_wrapper, comm_hook=default.bf16_compress_hook)
    )
    QUANTIZE_PER_TENSOR = _enum_member(
        partial(
            _ddp_comm_hook_wrapper, comm_hook=quantization.quantization_pertensor_hook
        )
    )
    QUANTIZE_PER_CHANNEL = _enum_member(
        partial(
            _ddp_comm_hook_wrapper, comm_hook=quantization.quantization_perchannel_hook
        )
    )
    POWER_SGD = _enum_member(
        partial(
            _powerSGD_comm_hook_wrapper,
            comm_hook=powerSGD.powerSGD_hook,
            matrix_approximation_rank=1,
        )
    )
    # Rank-2 PowerSGD can give a higher accuracy than the default rank-1 version,
    # but it runs slower and consumes more memory.
    POWER_SGD_RANK2 = _enum_member(
        partial(
            _powerSGD_comm_hook_wrapper,
            comm_hook=powerSGD.powerSGD_hook,
            matrix_approximation_rank=2,
        )
    )
    # Batching can lead to a faster training at the cost of accuracy.
    BATCHED_POWER_SGD = _enum_member(
        partial(
            _powerSGD_comm_hook_wrapper,
            comm_hook=powerSGD.batched_powerSGD_hook,
            matrix_approximation_rank=1,
        )
    )
    BATCHED_POWER_SGD_RANK2 = _enum_member(
        partial(
            _powerSGD_comm_hook_wrapper,
            comm_hook=powerSGD.batched_powerSGD_hook,
            matrix_approximation_rank=2,
        )
    )
    NOOP = _enum_member(
        partial(
            _ddp_comm_hook_wrapper,
            comm_hook=debugging.noop_hook,
        )
    )


def register_ddp_comm_hook(comm_hook_type: DDPCommHookType, model, state=None):
    """
    Register ``ddp_comm_hooks`` to DDP model.

    Registers the hooks of ``torch.distributed.algorithms.ddp_comm_hooks``
    to the DDP model. User can specify the type of hook as an enum
    ``DDPCommHookType`` type using ``comm_hook_type`` input. State input will
    be passed to the model.
    Uses Python comm hook implementations.

    Example::
        >>> # xdoctest: +SKIP
        >>> register_ddp_comm_hook(DDPCommHookType.FP16_COMPRESS, model, state)
    """
    comm_hook_type.value(model=model, state=state)
<|endoftext|>from dataclasses import dataclass
from typing import Any, no_type_check

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch.distributed.utils import _free_storage


@dataclass
class _AllreduceUpcastHookState:
    """
    State to manage DDP mixed precision in backward / gradient communication.

    This contains a weakref to the DDP module for access to reducer and process
    group, and a stream to run parameter and gradient upcasts.
    """

    ddp_weakref: Any
    upcast_stream: torch.Stream
    wait_for_stream_enqueued: bool = False


@no_type_check
def _reducer_allreduce_and_upcast_hook(
    hook_state: _AllreduceUpcastHookState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Perform allreduce in precision ``reduce_dtype``, upcast to prepare for optimizer.

    Performs allreduce in the reduced precision given by DDP's mixed precision
    reduce_dtype, and upcasts parameters and gradients to fp32 in preparation
    to run the optimizer.
    """
    ddp_weakref = hook_state.ddp_weakref
    reducer, process_group = ddp_weakref().reducer, ddp_weakref().process_group
    # Cast bucket if different than param_dtype.
    if (
        ddp_weakref().mixed_precision.param_dtype
        != ddp_weakref().mixed_precision.reduce_dtype
    ):
        # Cast bucket tensor to reduce_dtype
        bucket.set_buffer(
            bucket.buffer().to(ddp_weakref().mixed_precision.reduce_dtype)
        )
    fut = reducer._run_allreduce_hook(bucket)
    ret_fut = torch.futures.Future()
    stream = hook_state.upcast_stream
    with stream:
        fut.wait()
        bucket.buffer().div_(process_group.size())
        ret_fut.set_result(bucket.buffer())

        # Upcast parameters and gradients so optimizer step can run in fp32.
        for p in bucket.parameters():
            p.data = p._fp_param
            # free storage for mp param as it will be allocated again in next
            # forward pass.
            _free_storage(p._mp_param)
            p.grad.data = p.grad.to(p.data.dtype)

    # enqueue a callback to wait for this stream at end of backward
    def wait_for_stream_cb():
        torch.accelerator.current_stream().wait_stream(stream)
        # Remove post-backward hooks since they are re-installed in next
        # iteration, similar to FSDP.
        # Parameters that don't require grad still needed to be casted since
        # they may participate in computation. However, they would not be recast
        # by hook above as they don't have a grad hook installed, so cast them
        # back here.
        for _, p in ddp_weakref().module.named_parameters():
            if hasattr(p, "_ddp_mp_hook_state"):
                p._ddp_mp_hook_state[1].remove()
                delattr(p, "_ddp_mp_hook_state")
            if not p.requires_grad and not hasattr(p, "_ddp_ignored"):
                p.data = p._fp_param

        # reset for next backward pass
        hook_state.wait_for_stream_enqueued = False

    if not hook_state.wait_for_stream_enqueued:
        Variable._execution_engine.queue_callback(wait_for_stream_cb)
        # mark that the callback is enqueued
        hook_state.wait_for_stream_enqueued = True

    return ret_fut
<|endoftext|># mypy: allow-untyped-defs
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.utils as utils
from torch.utils._typing_utils import not_none as _not_none


__all__ = ["ModelAverager", "PeriodicModelAverager"]


class ModelAverager(ABC):
    r"""Base class for all model averagers.

    Args:
        process_group: The process group to be used for all-reduce.
                       If ``None``, the default process group, which
                       is created by :func:`torch.distributed.init_process_group`,
                       will be used. (default: ``None``)
    """

    def __init__(self, process_group: dist.ProcessGroup | None = None):
        self.process_group = (
            process_group if process_group is not None else _not_none(dist.group.WORLD)
        )
        self.step = 0

    @abstractmethod
    def average_parameters(self, params):
        raise NotImplementedError


class PeriodicModelAverager(ModelAverager):
    r"""
    Averages parameters periodically after the warm-up stage.

    This can be used for running `post-local SGD <https://arxiv.org/abs/1808.07217>`_,
    by running :class:`~torch.nn.DistributedDataParallel` (DDP)
    using the subgroups created by :meth:`~torch.distributed.new_subgroups`.

    Args:
        period (int): The number of steps per model averaging.
                      Usually the period should be greater than ``1`` to reduce the communication cost.
                      Otherwise, only DDP needs to be used.
        warmup_steps (int): The number of warm-up steps. During this stage,
                            model averaging is skipped.
        process_group: The process group to be used for all-reduce.
                       If ``None``, the default process group, which
                       is created by :func:`torch.distributed.init_process_group`,
                       will be used. (default: ``None``)

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> import torch
        >>> import torch.distributed as dist
        >>> import torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook as post_localSGD
        >>> import torch.distributed.algorithms.model_averaging.averagers as averagers
        >>> import torch.nn as nn
        >>>
        >>> dist.init_process_group("nccl", rank=rank, world_size=16)
        >>> torch.cuda.set_device(rank)
        >>> module = nn.Linear(1, 1, bias=False).cuda()
        >>> model = nn.parallel.DistributedDataParallel(
        >>>    module, device_ids=[rank], output_device=rank
        >>> )
        >>> # Register a post-localSGD communication hook.
        >>> state = PostLocalSGDState(process_group=None, subgroup=None, start_localSGD_iter=100)
        >>> model.register_comm_hook(state, post_localSGD_hook)
        >>>
        >>> # In the first 100 steps, run global gradient averaging like normal DDP at every step.
        >>> # After 100 steps, run model averaging every 4 steps.
        >>> # Note that ``warmup_steps`` must be the same as ``start_localSGD_iter`` used in ``PostLocalSGDState``.
        >>> averager = averagers.PeriodicModelAverager(period=4, warmup_steps=100)
        >>> for step in range(0, 200):
        >>>    optimizer.zero_grad()
        >>>    loss = loss_fn(output, labels)
        >>>    loss.backward()
        >>>    optimizer.step()
        >>>    # Will average model parameters globally every 4 steps. Thus,
        >>>    # inter-node communication only occurs every 4 iterations after
        >>>    # the initial ``warmup_steps`` period.
        >>>    averager.average_parameters(model.parameters())
    """

    def __init__(
        self, period, warmup_steps=0, process_group: dist.ProcessGroup | None = None
    ):
        super().__init__(process_group)
        if warmup_steps < 0:
            raise ValueError("Arg ``warmup_steps`` must be a non-negative number.")
        self.warmup_steps = warmup_steps
        if period < 1:
            raise ValueError("Arg ``period`` must be a positive value.")
        elif period == 1:
            warnings.warn(
                "When period is 1, no need to use model averaging because the communication cost "
                "of all-reducing parameters will be no less than the cost of all-reducing gradients "
                "by DistributedDataParallel in the backward pass. Therefore, only "
                "DistributedDataParallel should be used for this case.",
                stacklevel=2,
            )
        self.period = period

    def average_parameters(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, torch.nn.Parameter]],
    ):
        """
        Averages parameters or parameter groups of an optimizer if ``step`` is no less than ``warmup_steps``.

        Can be divided by ``period``, where ``step`` is increased by 1
        at each iteration in the training loop.
        Args:
            params: The parameters of a model or parameter groups of an optimizer.

        """
        if (
            self.step >= self.warmup_steps
            and (self.step - self.warmup_steps) % self.period == 0
        ):
            utils.average_parameters_or_parameter_groups(
                params, _not_none(self.process_group)
            )
        self.step += 1
<|endoftext|># mypy: allow-untyped-defs
import itertools
from collections.abc import Iterable, Iterator

import torch
import torch.distributed as dist

# The two imports below are not always available depending on the
# USE_DISTRIBUTED compile flag. Make sure they raise import error
# if we're trying to use them.
from torch.distributed import group, ProcessGroup


__all__ = [
    "average_parameters",
    "get_params_to_average",
    "average_parameters_or_parameter_groups",
]


def average_parameters(
    params: Iterator[torch.nn.Parameter], process_group: ProcessGroup
):
    """
    Averages all the given parameters.

    For allreduce efficiency, all the parameters are flattened into a contiguous buffer.
    Thus, it requires extra memory of the same size as the given parameters.
    """
    group_to_use = process_group if process_group is not None else group.WORLD
    # Do not update any parameter if not in the process group.
    if dist._rank_not_in_group(group_to_use):
        return

    params_it1, params_it2 = itertools.tee(params)
    # If the input parameters have different data types,
    # packing these parameters will trigger an implicit type up-casting.
    # The original parameter data types will be restored during the subsequent unpacking.
    flat_params = torch.cat([p.data.reshape(-1) for p in params_it1])
    flat_params /= dist.get_world_size(group_to_use)
    # Make sure the allreduce will not conflict with any other ongoing process group.
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()
    dist.all_reduce(flat_params, group=group_to_use)

    offset = 0
    for p in params_it2:
        p.data = flat_params[offset : offset + p.numel()].view_as(p).type_as(p)
        offset += p.numel()


def get_params_to_average(
    params: Iterable[torch.nn.Parameter] | Iterable[dict[str, torch.nn.Parameter]],
):
    """
    Return a list of parameters that need to average.

    This filters out the parameters that do not contain any gradients.
    Args:
        params: The parameters of a model or parameter groups of an optimizer.
    """
    filtered_params = []
    for param in params:
        if isinstance(param, torch.nn.Parameter):
            # model.parameters() input
            param_data = param
            if param_data.grad is not None:
                filtered_params.append(param_data)
        elif isinstance(param, dict):
            # optimizer.param_groups input
            for param_data in param["params"]:
                if param_data.grad is not None:
                    filtered_params.append(param_data)
        else:
            raise NotImplementedError(
                f"Parameter input of type {type(param)} is not supported"
            )
    return filtered_params


def average_parameters_or_parameter_groups(
    params: Iterable[torch.nn.Parameter] | Iterable[dict[str, torch.nn.Parameter]],
    process_group: ProcessGroup,
):
    """Averages parameters of a model or parameter groups of an optimizer."""
    average_parameters(iter(get_params_to_average(params)), process_group)
<|endoftext|># mypy: allow-untyped-defs
# Copyright 2022 Cruise LLC
import logging
import warnings
from collections import OrderedDict
from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.distributed.algorithms.model_averaging.utils as utils


logger = logging.getLogger(__name__)


class HierarchicalModelAverager(averagers.ModelAverager):
    r"""
    Runs hierarchical model averaging (`hierarchical SGD <https://arxiv.org/pdf/2010.12998.pdf>`_).

    Process groups of different sizes are organized in a hierarchy, and they average parameters
    by using different periods concurrently after the warm-up stage.
    This is an extension of :class:`~torch.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager`
    that supports `post-local SGD <https://arxiv.org/abs/1808.07217>`_, which essentially only supports
    a two-level hierarchy: the intra-machine level and the global level, where the intra-machine
    level is usually embedded in :meth:`~torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook`.
    Similarly, the process groups within this class do not have such an intra-machine process
    subgroup, which should be embedded by the post-local SGD communication hook instead.

    Args:
        period_group_size_dict: An ordered dict mapping keys of model averaging period to
                                process group size, used for initializing process groups of
                                different sizes in a hierarchy to average parameters concurrently.
                                Particularly, at each iteration, there will be at most a single
                                process group that runs averaging -- the period of such group should
                                have the largest period which the current step can be divided by.
                                For example, if the dict has three keys: 2, 4, and 8,
                                then this means totally three process groups will be created to
                                average parameters every 2, 4, and 8 iterations, respectively.
                                At the 4th iteration, only the second process group will run
                                averaging, because the first process group should be a
                                subset of the second process group, and no need to execute the first
                                process group redundantly.
                                On the other hand, the third process group can only be triggered
                                every 8 iterations, so it will not be triggered at the 4th iteration.
        warmup_steps (int): The number of warm-up steps. During this stage, model averaging is skipped.
        process_group (ProcessGroup, optional): The overall process group containing all the processes that runs model averaging.
                                                If ``None``, the default process group, which is created
                                                by :func:`torch.distributed.init_process_group`, will be used.
                                                (default: ``None``)

    Example::
        >>> # xdoctest: +SKIP('undefined rank')
        >>> from collections import OrderedDict
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
        >>>     PostLocalSGDState,
        >>>     post_localSGD_hook,
        >>> )
        >>> import torch.distributed.algorithms.model_averaging.hierarchical_model_averager as hierarchicalSGD
        >>> import torch.nn as nn
        >>>
        >>> dist.init_process_group("nccl", rank=rank, world_size=16)
        >>> torch.cuda.set_device(rank)
        >>> module = nn.Linear(1, 1, bias=False).to(rank)
        >>> model = nn.parallel.DistributedDataParallel(
        >>>    module, device_ids=[rank], output_device=rank
        >>> )
        >>> # Register a post-localSGD communication hook.
        >>> # Assume that each machine has 4 GPUs, then each intra-machine subgroup has a size of 4.
        >>> subgroup, _ = dist.new_subgroups()
        >>> state = PostLocalSGDState(process_group=None, subgroup=subgroup, start_localSGD_iter=100)
        >>> model.register_comm_hook(state, post_localSGD_hook)
        >>>
        >>> # Average parameters among each group of 8 processes every 4 iterations, and among all
        >>> # the 16 processes every 16 iterations.
        >>> averager = hierarchicalSGD.HierarchicalModelAverager(
        >>>     period_group_size_dict=OrderedDict([(4, 8), (16, 16)]), warmup_steps=100)
        >>> # Note that ``warmup_steps`` must be the same as ``start_localSGD_iter`` used in ``PostLocalSGDState``.
        >>> # In the first 100 steps, run global gradient averaging like normal DDP at every step.
        >>> # After 100 steps, run model averaging at two levels.
        >>> for step in range(0, 200):
        >>>    optimizer.zero_grad()
        >>>    loss = loss_fn(output, labels)
        >>>    loss.backward()
        >>>    optimizer.step()
        >>>    # Average parameters after ``optimizer.step()``.
        >>>    # Thus, the inter-node communication only occurs periodically after ``warmup_steps``.
        >>>    averager.average_parameters(model.parameters())

    .. warning ::
        The last group size in the dict must be the size of the provided ``process_group``,
        which indicates model averaging at the highest level of the hierarchy.
        If ``process_group`` is not provided, then the last group size should be equal to the world size.

    .. warning ::
        `HierarchicalModelAverager` is experimental and subject to change.
    """

    def __init__(self, period_group_size_dict=None, warmup_steps=0, process_group=None):
        super().__init__(process_group)
        if not period_group_size_dict:
            raise ValueError("Arg ``period_group_size_dict`` must not be empty.")
        self._periods = list(period_group_size_dict.keys())
        if self._periods[0] <= 0:
            raise ValueError(
                "The minimum period in arg ``period_group_size_dict`` must be a positive value."
            )
        elif self._periods[-1] == 1:
            warnings.warn(
                "When the maximum period in arg ``period_group_size_dict`` is 1, "
                "no need to use model averaging because the communication cost "
                "of all-reducing parameters will be no less than the cost of all-reducing gradients "
                "by DistributedDataParallel in the backward pass. Therefore, only "
                "DistributedDataParallel should be used for this case.",
                stacklevel=2,
            )
        overall_group_size = dist.get_world_size(group=self.process_group)
        if list(period_group_size_dict.values())[-1] != overall_group_size:
            raise ValueError(
                f"The last value in arg ``period_group_size_dict`` {list(period_group_size_dict.values())[-1]} "
                f"must be equal to the size of arg ``process_group`` {overall_group_size}."
            )

        self.period_process_group_dict = OrderedDict()
        logger.info("Model averaging hierarchy:")
        for period, group_size in period_group_size_dict.items():
            logger.info(
                "\tEach group that has %s processes average parameters every %s iterations, "
                "if no higher-level averaging.",
                group_size,
                period,
            )
            if group_size != overall_group_size:
                self.period_process_group_dict[period], _ = dist.new_subgroups(
                    group_size=group_size, group=self.process_group
                )
            else:
                self.period_process_group_dict[period] = self.process_group

        if warmup_steps < 0:
            raise ValueError("Arg ``warmup_steps`` must be a non-negative number.")
        self.warmup_steps = warmup_steps

    def _find_process_group(self):
        """
        Return a process group as the value of a ``period_process_group_dict`` entry.

        If ``step`` can be divided by multiple periods in the keys of ``period_process_group_dict``,
        then the returned process group is the one corresponding to the largest period,
        since this process group will be used for averaging parameters at this ``step``.
        Returns ``None`` if not found.
        """
        for period in reversed(self._periods):
            if self.step % period == 0:
                return self.period_process_group_dict[period]
        return None

    def average_parameters(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, torch.nn.Parameter]],
    ):
        """
        Averages parameters or parameter groups of an optimizer.

        Averaging only occurs if ``step`` is no less than ``warmup_steps``
        and it can be divided by a period in the keys of ``period_process_group_dict``,
        where ``step`` is increased by 1 at each iteration in the training loop.
        If ``step`` can be divided by multiple periods in the keys of ``period_process_group_dict``,
        only the largest period is used, and the corresponding process group is used for averaging parameters.
        Args:
            params: The parameters of a model or parameter groups of an optimizer.
        """
        if self.step >= self.warmup_steps:
            group = self._find_process_group()
            if group is not None:
                utils.average_parameters_or_parameter_groups(params, group)
        self.step += 1
<|endoftext|># mypy: allow-untyped-defs
import inspect
from abc import ABC, abstractmethod

from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import (
    _hook_then_optimizer,
    _OptimizerHookState,
)
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.optim import as_functional_optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer


# Contains the mappings between the regular and overlapped optimizer types.
_registered_overlapped_optims: dict[type, type] = {}


def register_overlapped(optim_cls):
    def decorator(target_overlapped_optim_cls):
        if target_overlapped_optim_cls in _registered_overlapped_optims:
            raise ValueError(
                f"{target_overlapped_optim_cls} already registered with optim_cls "
                f"{_registered_overlapped_optims[optim_cls]} {optim_cls}, trying to"
                f"re-register it for {optim_cls} is not supported."
            )
        _registered_overlapped_optims[optim_cls] = target_overlapped_optim_cls
        return target_overlapped_optim_cls

    return decorator


class OverlappedOptimizer(ABC):
    def __init__(self, optim_cls: type) -> None:
        """
        Initialize the OverlappedOptimizer.

        Overlappedoptimizer is a base class that child classes can implement to
        specify how different optimizers will register themselves with DDP.
        """
        self.optim_cls = optim_cls

    @abstractmethod
    def register_ddp(self, ddp: DistributedDataParallel) -> None:
        """Registers the overlapped optimizer with DDP."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped DDP."
        )

    @abstractmethod
    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        """Registers the overlapped optimizer with FSDP."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped FSDP."
        )


@register_overlapped(Optimizer)
class _OverlappedStandardOptimizer(OverlappedOptimizer):
    """Overlaps a regular ``Optimizer``."""

    def __init__(self, optim_cls: type, params, *optim_args, **optim_kwargs) -> None:
        super().__init__(optim_cls)
        f_optim = as_functional_optim(self.optim_cls, *optim_args, **optim_kwargs)
        self._opt_hook_state = _OptimizerHookState(f_optim, params)

    def register_ddp(self, ddp_inst: DistributedDataParallel):
        # NOTE: using a custom communication hook and fused optimizer is not
        # yet supported.
        ddp_inst.register_comm_hook(  # type: ignore[operator]
            None,  # wrapped hook state
            _hook_then_optimizer(allreduce_hook, self._opt_hook_state),
        )

    # TODO: register_fsdp once FSDP supports communication hook.
    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        """Register the overlapped optimizer with FSDP."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped FSDP."
        )


def _as_overlapped_optim(optim_cls: type, params, *args, **kwargs):
    """Return a new ``OverlappedOptimizer`` instance that supports ``optim_cls``."""
    for clz in inspect.getmro(optim_cls):
        try:
            return _registered_overlapped_optims[clz](
                optim_cls, params, *args, **kwargs
            )
        except KeyError:
            pass

    # Fallback to standard overlapped optimizer, which will raise errors if user
    # is attempting to use an unsupported optimizer.
    return _OverlappedStandardOptimizer(optim_cls, params, *args, **kwargs)
<|endoftext|># mypy: allow-untyped-defs
import functools
from enum import Enum

import torch
import torch.distributed as dist


TORCH_HALF_MIN = torch.finfo(torch.float16).min
TORCH_HALF_MAX = torch.finfo(torch.float16).max


class DQuantType(Enum):
    """
    Different quantization methods for auto_quantize API are identified here.

    auto_quantize API currently supports fp16 and bfp16 methods.
    """

    FP16 = ("fp16",)
    BFP16 = "bfp16"

    def __str__(self) -> str:
        return self.value


def _fp32_to_fp16_with_clamp(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor, TORCH_HALF_MIN, TORCH_HALF_MAX).half()


def _quantize_tensor(tensor, qtype):
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"_quantize_tensor expecting torch.Tensor as input but found {type(tensor)}"
        )
    if qtype == DQuantType.FP16:
        return _fp32_to_fp16_with_clamp(tensor)
    elif qtype == DQuantType.BFP16:
        return torch.ops.quantization._FloatToBfloat16Quantized(tensor)
    else:
        raise RuntimeError(f"Quantization type {qtype} is not supported")


def _quantize_tensor_list(tensor_list, qtype):
    if not isinstance(tensor_list, list) or not all(
        isinstance(p, torch.Tensor) for p in tensor_list
    ):
        raise RuntimeError(
            f"_quantize_tensor_list expecting list of torch.Tensor as input but found {type(tensor_list)}"
        )
    quantized_tensor_list = [_quantize_tensor(t, qtype) for t in tensor_list]
    return quantized_tensor_list


def _dequantize_tensor(tensor, qtype, quant_loss=None):
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"_dequantize_tensor expecting torch.Tensor as input but found {type(tensor)}"
        )
    if qtype == DQuantType.FP16:
        if tensor.dtype != torch.float16:
            raise RuntimeError(
                f"tensor dtype is {tensor.dtype} while expected to be FP16."
            )
        elif tensor.dtype == torch.float16 and quant_loss is None:
            return tensor.float()
        else:
            # pyrefly: ignore [unsupported-operation]
            return tensor.float() / quant_loss
    elif qtype == DQuantType.BFP16:
        if tensor.dtype != torch.float16:
            raise RuntimeError(
                f"tensor dtype is {tensor.dtype} while expected to be FP16."
            )
        else:
            return torch.ops.quantization._Bfloat16QuantizedToFloat(tensor)
    else:
        raise RuntimeError(f"Quantization type {qtype} is not supported")


def _dequantize_tensor_list(tensor_list, qtype, quant_loss=None):
    if not isinstance(tensor_list, list) or not all(
        isinstance(p, torch.Tensor) for p in tensor_list
    ):
        raise RuntimeError(
            f"_dequantize_tensor_list expecting list of torch.Tensor as input but found {type(tensor_list)}"
        )
    dequantized_tensor_list = [_dequantize_tensor(t, qtype) for t in tensor_list]
    return dequantized_tensor_list


def auto_quantize(func, qtype, quant_loss=None):
    """
    Quantize the input tensors, choose the precision types, and pass other necessary arguments and then dequantizes the output.

    Currently it only supports:
        . FP16 and BFP16 quantization method supported for gloo and nccl backends
        . all_gather, all_to_all collective ops
    Note: BFP16 only supports 2D tensors.
    Args:
        func (Callable): A function representing collective operations.
        qtype (QuantType): Quantization method
        quant_loss (float, optional): This can be used to improve accuracy in the dequantization.
    Returns:
        (Callable): the same collective as func but enables automatic quantization/dequantization.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        group = kwargs.get("group")
        async_op = kwargs.get("async_op", False)
        if async_op is True:
            raise RuntimeError("The async_op=True mode is not supported yet.")
        if func is dist.all_gather:
            tensors = args[0]
            input_tensors = _quantize_tensor(args[1], qtype)
            out_tensors = _quantize_tensor_list(tensors, qtype)
            dist.all_gather(out_tensors, input_tensors, group=group, async_op=async_op)
            for i, t in enumerate(
                _dequantize_tensor_list(out_tensors, qtype, quant_loss=quant_loss)
            ):
                tensors[i] = t

        elif func is dist.all_to_all:
            tensors = args[0]
            input_tensors = _quantize_tensor_list(args[1], qtype)
            out_tensors = _quantize_tensor_list(tensors, qtype)
            dist.all_to_all(out_tensors, input_tensors, group=group, async_op=async_op)
            for i, t in enumerate(
                _dequantize_tensor_list(out_tensors, qtype, quant_loss=quant_loss)
            ):
                tensors[i] = t

        elif func is dist.all_to_all_single:
            tensors = args[0]
            out_splits = kwargs.get("out_splits")
            in_splits = kwargs.get("in_splits")
            # Quantizing the input/output tensor
            input_tensors = _quantize_tensor(args[1], qtype)
            out_tensors = _quantize_tensor(tensors, qtype)
            dist.all_to_all_single(
                out_tensors, input_tensors, out_splits, in_splits, group=group
            )
            for i, t in enumerate(
                _dequantize_tensor(out_tensors, qtype, quant_loss=quant_loss)
            ):
                tensors[i] = t
        else:
            raise RuntimeError(f"The collective op {func} is not supported yet")

    return wrapper
<|endoftext|># mypy: allow-untyped-defs
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from enum import auto, Enum
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch.autograd.graph import save_on_cpu
from torch.distributed.utils import _pack_kwargs, _replace_by_prefix, _unpack_kwargs
from torch.utils._typing_utils import copy_method_params
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint


_CHECKPOINT_WRAPPED_MODULE = "_checkpoint_wrapped_module"
_CHECKPOINT_PREFIX = _CHECKPOINT_WRAPPED_MODULE + "."


class CheckpointImpl(Enum):
    REENTRANT = auto()
    NO_REENTRANT = auto()


class ActivationWrapper(torch.nn.Module, ABC):
    """
    Base class for Activation Checkpoint and Activation Offload.

    Not meant to be instantiated directly.
    """

    def __init__(self, mod):
        super().__init__()
        self._checkpoint_wrapped_module = mod
        # state_dict post hook to remove prefix to allow loading into a
        # non-checkpoint wrapped module.
        self._register_state_dict_hook(self._post_state_dict_hook)
        # load_state_dict pre-hook to allow loading back into
        # checkpoint-wrapped module.
        self.register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise ValueError("Subclasses should implement forward().")

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._checkpoint_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self._checkpoint_wrapped_module.__getitem__(key)  # type: ignore[operator]

    @copy_method_params(torch.nn.Module.named_parameters)
    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """
        Override :meth:`named_parameters()` to intercept parameter names.

        remove all occurrences of ``_CHECKPOINT_PREFIX``.
        """
        for param_name, param in super().named_parameters(*args, **kwargs):
            yield param_name.replace(_CHECKPOINT_PREFIX, ""), param

    @staticmethod
    def _post_state_dict_hook(
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> dict[str, Any]:
        """
        _post_state_dict_hook() is called after the state_dict() of this FSDP module is executed.

        For ``checkpoint_wrapper``, it will strip checkpoint-wrapped module prefix,
        so that this module can be loaded into non-checkpointed modules.
        It would still be able to be loaded into checkpoint-wrapped modules as this class,
        adds the prefix back before loading the state_dict.
        """
        _replace_by_prefix(state_dict, f"{prefix}{_CHECKPOINT_PREFIX}", prefix)
        return state_dict

    @staticmethod
    def _pre_load_state_dict_hook(
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()`` is called.

        For ``checkpoint_wrapper``, it will add back the module
        prefix so that non-checkpointed modules can be loaded into
        checkpoint_wrapper modules properly.
        """
        _replace_by_prefix(state_dict, prefix, prefix + f"{_CHECKPOINT_PREFIX}")


class OffloadWrapper(ActivationWrapper):
    def forward(self, *args, **kwargs):
        with save_on_cpu(pin_memory=True):
            return self._checkpoint_wrapped_module(*args, **kwargs)


class CheckpointWrapper(ActivationWrapper):
    """
    An ``nn.Module`` that wraps another ``nn.Module`` with checkpointing.

    Note that this module is not meant to be used directly but instead,
    it is to be used through the ``checkpoint_wrapper`` function.
    """

    def __init__(
        self,
        mod: torch.nn.Module,
        checkpoint_impl: CheckpointImpl = CheckpointImpl.NO_REENTRANT,
        checkpoint_fn=None,
        **checkpoint_fn_kwargs,
    ):
        super().__init__(mod)
        self.checkpoint_impl = checkpoint_impl
        if checkpoint_fn is None:
            # use torch.utils.checkpoint
            self.checkpoint_fn = partial(
                torch_utils_checkpoint,
                use_reentrant=(self.checkpoint_impl == CheckpointImpl.REENTRANT),
                **checkpoint_fn_kwargs,
            )
        else:
            # Construct user-specified checkpoint function.
            self.checkpoint_fn = partial(
                checkpoint_fn,
                **checkpoint_fn_kwargs,
            )

    def forward(self, *args, **kwargs):
        # Support keyword arguments for reentrant checkpoint. Note that this
        # only works if user has specified self.checkpoint_impl and is not
        # using their own custom checkpoint_fn.
        if self.checkpoint_impl == CheckpointImpl.REENTRANT and kwargs != {}:
            # Pack the args and kwargs
            flat_args, kwarg_keys = _pack_kwargs(*args, **kwargs)

            # Function that only takes (packed) args, but can unpack them
            # into the original args and kwargs for the checkpointed
            # function, and runs that function.
            def my_function(*inputs):
                # unpack back into args and kwargs
                unpacked_args, unpacked_kwargs = _unpack_kwargs(inputs, kwarg_keys)
                # run original module
                return self._checkpoint_wrapped_module(
                    *unpacked_args, **unpacked_kwargs
                )

            # Pass the function that only takes packed args into reentrant
            # checkpoint API.
            return self.checkpoint_fn(  # type: ignore[misc]
                my_function,
                *flat_args,
            )
        else:
            return self.checkpoint_fn(  # type: ignore[misc]
                self._checkpoint_wrapped_module, *args, **kwargs
            )


def offload_wrapper(module: torch.nn.Module) -> torch.nn.Module:
    """
    Wrap a module for activation offloading to CPU.

    Offloads intermediate activations to the CPU for modules wrapped with this function.
    Wrappers with activation offload can be composed with ones that do recomputation-based
    checkpoint to trade off increased compute versus increased CPU
    memory usage and additional H2D transfers.

    Usage::
        offloaded_module = offload_wrapper(module)
        outputs = checkpointed_module(inputs)
    Args:
        module (nn.Module):
            The module to be wrapped
    Returns:
        (nn.Module):
            Wrapped module
    """
    return OffloadWrapper(module)


def checkpoint_wrapper(
    module: torch.nn.Module,
    checkpoint_impl: CheckpointImpl = CheckpointImpl.NO_REENTRANT,
    checkpoint_fn=None,
    **checkpoint_fn_kwargs,
) -> torch.nn.Module:
    """
    Wrap a module for activation checkpointing.

    If the module is wrapped with this function, all subsequent calls to the module will,
    automatically perform checkpointing without the user having to explicitly call ``checkpoint`` function.

    Usage::
        checkpointed_module = checkpoint_wrapper(module)
        outputs = checkpointed_module(inputs)
    Args:
        module (nn.Module):
            The module to be wrapped
        checkpoint_impl (Optional[CheckpointImpl]):
            The checkpointing implementation to use. Note that this will only
            be passed into the ``torch.utils.checkpoint.checkpoint``
            implementation, and is ignored if a custom ``checkpoint_fn`` is
            specified. Note that for implementations using reentrant checkpoint
            from ``torch.utils.checkpoint``, keyword arguments will only be
            supported if ``checkpoint_impl`` is passed as ``CheckpointImpl.REENTRANT`.
        checkpoint_fn (Optional[Callable]):
            Functional checkpoint implementation to use. If this is specified,
            it will be used over the default ``torch.utils.checkpoint.checkpoint``
            implementation and the `checkpoint_impl` argument will be ignored.
        **checkpoint_fn_kwargs: (Dict[str, Any]): Keyword arguments to pass into `checkpoint_fn`.

    Returns:
        (nn.Module):
            Wrapped module
    """
    return CheckpointWrapper(
        module,
        checkpoint_impl,
        checkpoint_fn,
        **checkpoint_fn_kwargs,
    )


def apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=lambda _: True,
    auto_wrap_policy: Callable[[nn.Module, bool, int], bool] | None = None,
):
    """
    Apply :func:`checkpoint_wrapper` to modules within `model` based on a user-defined configuration.

    For each module within `model`, the `check_fn` is used to decide
    whether `module` should be wrapped with :func:`checkpoint_wrapper` or not.

    Note::
        This function modifies `model` in place and replaces appropriate layers with
        their checkpoint-wrapped modules.
    Note::
        This function will not wrap the overall root module. If this is needed, please directly use
        :func:`checkpoint_wrapper` or :func:`offload_wrapper`.
    Usage::
        model = nn.Sequential(
            nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10)
        )
        check_fn = lambda l: isinstance(l, nn.Linear)
        # checkpoint activations
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)
        # Or offload activations to CPU
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=offload_wrapper, check_fn=check_fn)
    Args:
        model (nn.Module):
            The model whose submodules should be wrapped with activation checkpointing.
        checkpoint_wrapper_fn (Optional[Callable[nn.Module]])
            A ``Callable`` which will wrap modules
        check_fn (Optional[Callable[nn.Module, nn.Module]])
            A lambda function which will be passed each child submodule of ``model`` and returns
            ``True`` or ``False`` depending on whether the submodule should be wrapped.
        auto_wrap_policy (Optional[Callable[[nn.Module, bool, int], bool]]): A policy to wrap model's
            submodules with AC. Note that if this is specified, it takes precedence over ``check_fn``.
    Returns: None (`model` is modified inplace)
    """
    # TODO: Importing inside function to avoid circular import issue between FSDP and
    # checkpoint_wrapper. This can be resolved once wrap() APIs are decoupled from FSDP code.
    from torch.distributed.fsdp._wrap_utils import _construct_wrap_fn, _post_order_apply
    from torch.distributed.fsdp.wrap import (
        _Policy,
        _recursive_wrap,
        lambda_auto_wrap_policy,
    )

    policy = (
        auto_wrap_policy
        if auto_wrap_policy is not None
        else partial(lambda_auto_wrap_policy, lambda_fn=check_fn)
    )
    if not callable(policy):
        if not isinstance(policy, _Policy):
            raise ValueError(
                f"Expected {policy} to be callable or be a pre-defined wrap policy"
            )
        target_module_to_kwargs = policy._run_policy(
            model, ignored_modules=set(), root_kwargs={}
        )
        wrap_fn = _construct_wrap_fn(
            model, target_module_to_kwargs, checkpoint_wrapper_fn
        )
        _post_order_apply(model, wrap_fn)
        return

    _recursive_wrap(
        module=model,
        auto_wrap_policy=policy,  # type: ignore[arg-type]
        wrapper_cls=checkpoint_wrapper_fn,
        ignored_modules=set(),
        ignored_params=set(),
        only_wrap_children=True,
    )
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
import logging
import warnings
from collections.abc import Sequence
from typing import cast

import torch
import torch.distributed as dist
import torch.distributed.tensor._api as dtensor
import torch.distributed.tensor._random as random
from torch._library.utils import fill_defaults
from torch._logging import LazyString
from torch._prims.rng_prims import run_dtensor_rng_op
from torch.distributed._functional_collectives import _are_we_tracing
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._nonlinear_redux import argminmax_handler
from torch.distributed.tensor._op_schema import (
    OpInfo,
    OpSchema,
    OutputSharding,
    OutputSpecType,
)
from torch.distributed.tensor._random import is_rng_supported_mesh
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor._sharding_prop import ShardingPropagator
from torch.distributed.tensor._tp_conv import (
    convolution_backward_handler,
    convolution_handler,
)
from torch.distributed.tensor._utils import (
    _format_implicit_redistribution_msg,
    ExplicitRedistributionContext,
    try_find_mesh_from_args,
)
from torch.distributed.tensor.placement_types import Partial, Placement, Replicate
from torch.utils._debug_mode import get_active_debug_mode
from torch.utils._python_dispatch import return_and_correct_aliasing


try:
    from torch.utils import _cxx_pytree as pytree
except ImportError:
    from torch.utils import _pytree as pytree  # type: ignore[no-redef]

aten = torch.ops.aten
logger = logging.getLogger(__name__)

# The C++ DTensor dispatch fast path caches whether debug logging is
# enabled.  Wrap setLevel so the cached flag is reset automatically.
_orig_setLevel = logger.setLevel


def _setLevel_and_reinit(level: int) -> None:
    _orig_setLevel(level)
    torch._C._reinit_DTensor_dispatch_logger()


logger.setLevel = _setLevel_and_reinit  # type: ignore[method-assign]


def _ignore_fresh_unbacked_symbols_for_dtensor_tracing(
    output_spec: object,
) -> contextlib.AbstractContextManager[None]:
    if not _are_we_tracing() or output_spec is None:
        return contextlib.nullcontext()

    from torch.fx.experimental.symbolic_shapes import (
        _ignore_fresh_unbacked_symbols_tls_context,
    )

    # DTensor exposes the global output contract through output_spec. Fresh
    # local fake symbols created only while redistributing or computing the
    # per-rank tensor are implementation details and may not appear in the
    # returned DTensor object.
    return _ignore_fresh_unbacked_symbols_tls_context()


def _as_strided_permutation(
    tensor: "dtensor.DTensor",
    size: Sequence[int],
    stride: Sequence[int],
) -> list[int] | None:
    """Return dims such that ``tensor.permute(dims)`` has the requested layout.

    AOTAutograd regenerates an output that aliases an input by calling
    as_strided on the input, which a sharded tensor cannot answer in general --
    it has no single global storage to index into. A permutation of the base
    dims (transpose, t, permute, Tensor.T) is the exception: it maps onto
    aten.permute, which does have a sharding strategy.

    Returns None when the request is not a permutation, or when two dims have
    the same (size, stride) and the permutation is therefore ambiguous -- the
    layouts would be identical but the placements need not be.
    """
    if len(size) != tensor.dim():
        return None

    base = list(zip(tensor.size(), tensor.stride()))
    dims = []
    used = [False] * len(base)
    for target in zip(size, stride):
        candidates = [
            i for i, b in enumerate(base) if not used[i] and tuple(b) == tuple(target)
        ]
        if len(candidates) != 1:
            return None
        used[candidates[0]] = True
        dims.append(candidates[0])
    return dims


def as_strided_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
):
    args, kwargs = fill_defaults(op_call._schema, args, kwargs)
    if kwargs:
        raise AssertionError
    tensor, size, stride, storage_offset = args
    if storage_offset is None or tensor.storage_offset() == storage_offset:
        if tensor.size() == tuple(size) and tensor.stride() == tuple(stride):
            return torch.ops.aten.alias.default(tensor)
        dims = _as_strided_permutation(tensor, size, stride)
        if dims is not None:
            return torch.ops.aten.permute.default(tensor, dims)
    raise RuntimeError("as_strided not supported with DTensor")


def is_same_size_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> bool:
    lhs = cast(torch.Tensor, args[0])
    rhs = cast(torch.Tensor, args[1])
    return lhs.shape == rhs.shape


def is_pinned_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> bool:
    tensor = cast(dtensor.DTensor, args[0])
    return tensor._local_tensor.is_pinned()


def found_inf_reduce_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    local_tensor_args = pytree.tree_unflatten(
        cast(list[object], op_info.local_args),
        op_info.args_tree_spec,  # type: ignore[arg-type]
    )
