"""
.. autofunction:: record
.. autofunction:: query
.. autoclass:: QueryInfo
"""

import sys
import os
import logging
import sqlite3
import numpy as np
import loopy as lp

from dataclasses import dataclass
from typing import (TYPE_CHECKING, Optional, Union, Callable,
                    Tuple, FrozenSet, Any, Dict)
from pyrsistent.typing import PMap as PMapT
from feinsum.einsum import FusedEinsum, INT_CLASSES, SizeParam

logger = logging.getLogger(__name__)


if TYPE_CHECKING or getattr(sys, "FEINSUM_BUILDING_SPHINX_DOCS", False):
    # avoid making pyopencl a hard dep.
    import pyopencl as cl


# transform: (t_unit, insn_match, kernel_name)
TransformT = Callable[["lp.TranslationUnit", Optional[Any], Optional[str]],
                      "lp.TranslationUnit"]
FallbackT = Union[str, TransformT]


DEFAULT_FALLBACKS = ()

DEFAULT_TRANSFORM_ARCHIVE = os.path.join(os.path.dirname(__file__),
                                         "../../data/transform_archive_v0.db")


def _get_clbl_from_string(transform_src: str) -> TransformT:

    result_dict: Dict[Any, Any] = {}
    exec(compile(transform_src, "<feinsum transform code>", "exec"), result_dict)
    # To make the transform code debuggable by pudb
    result_dict["_MODULE_SOURCE_CODE"] = transform_src

    clbl = result_dict["transform"]
    if not callable(result_dict.get("transform")):
        raise ValueError("Provided transform source does not"
                         " define callable named 'transform'.")
    return clbl  # type: ignore[no-any-return]


def _get_value_to_dtype_for_db(einsum: FusedEinsum) -> str:
    return ("["
            + ", ".join(f"{val}: {dtype.name}"
                           for val, dtype in sorted(einsum
                                                    .value_to_dtype
                                                    .items()))
            + "]")


def _get_index_to_length_for_db(einsum: FusedEinsum) -> str:
    return "[" + ", ".join(f"{einsum.index_names[k]}: {v}"
                           for k, v in einsum.index_to_dim_length().items()
                           if isinstance(v, INT_CLASSES)) + "]"


def _get_use_matrix_for_db(einsum: FusedEinsum) -> str:
    def _stringify_use_row(use_row: Tuple[FrozenSet[str], ...]) -> str:
        normalized_use_row = [sorted({use
                                      for use in uses})
                              for uses in use_row]
        return ("["
                + ", ".join("[" + ", ".join(uses) + "]"
                            for uses in normalized_use_row)
                + "]")

    return "[" + ",\n".join(_stringify_use_row(use_row)
                            for use_row in einsum.use_matrix) + "]"


def _get_cl_version_for_db(cl_device: "cl.Device") -> str:
    # TODO: needs to consider more things into account
    return f"{cl_device.vendor}-{cl_device.driver_version}"


def _get_op_info_for_db(einsum: FusedEinsum, long_dim_length: int) -> str:
    from feinsum.measure import _get_giga_ops_from_einsum
    from pymbolic.mapper.evaluator import evaluate_to_float

    eval_context = {dim.name: long_dim_length
                    for dim in einsum.index_to_dim_length().values()
                    if isinstance(dim, SizeParam)}
    dtype_to_ops = {k: evaluate_to_float(v, eval_context)
                    for k, v in _get_giga_ops_from_einsum(einsum).items()}
    return "\n".join(f"{k.name}: {v}"
                     for k, v in dtype_to_ops.items())


def _get_log_str_for_run(einsum: FusedEinsum,
                         runtime: float,
                         device: "cl.Device",
                         long_dim_length: int) -> str:

    try:
        from tabulate import tabulate
    except ImportError:
        raise ImportError("`tabulate` is need for pretty printing."
                          " Install via `pip install tabulate`.")

    from feinsum.measure import (_get_giga_ops_from_einsum,
                                 _get_footprint_gbytes)
    from feinsum.data.device_info import (DEV_TO_PEAK_GFLOPS,
                                          DEV_TO_PEAK_BW)
    from pymbolic.mapper.evaluator import evaluate_to_float

    perf_table = [["Dtype", "Measured GOps/s", "Roofline GOps/s"]]

    eval_context = {dim.name: long_dim_length
                    for dim in einsum.index_to_dim_length().values()
                    if isinstance(dim, SizeParam)}
    dtype_to_ops = {k: evaluate_to_float(v, eval_context)
                    for k, v in _get_giga_ops_from_einsum(einsum).items()}
    ngbs = _get_footprint_gbytes(einsum, long_dim_length=long_dim_length)

    for dtype, ops in sorted(dtype_to_ops.items(),
                             key=lambda x: x[0].itemsize):
        roofline_flops = (ops
                          / max(ops/DEV_TO_PEAK_GFLOPS[device.name][dtype.name],
                                ngbs/DEV_TO_PEAK_BW[device.name]))
        perf_table.append([dtype.name,
                           f"{(ops/runtime):.1f}",
                           f"{roofline_flops:.1f}"])

    return tabulate(perf_table, tablefmt="fancy_grid")


def _get_cl_device_name_for_db(cl_device: "cl.Device") -> str:
    dev_name = cl_device.name
    assert isinstance(dev_name, str)
    return (dev_name
            .replace(" ", "_")
            .replace("-", "_")
            .replace("@", "AT")
            .replace("(", "_")
            .replace(")", "_")
            .replace(".", "DOT")
            )


def record(einsum: FusedEinsum,
           cl_ctx: "cl.Context",
           *,
           transform_str: Optional[str] = None,
           transform_file_path: Optional[str] = None,
           authors: str,
           remarks: str = "",
           database: str = DEFAULT_TRANSFORM_ARCHIVE,
           long_dim_length: int = 50_000,
           log_performance_data: bool = True,
           ) -> None:
    """
    :param log_performance_data: If *True* will log the run results via
        :mod:`logging`.
    """

    from feinsum.measure import timeit
    from feinsum.codegen.loopy import generate_loopy
    from feinsum.normalization import normalize_einsum
    einsum = normalize_einsum(einsum)

    # TODO: Instead of taking in a long_dim_length, should allow setting each
    # parameter its value.
    if (transform_str is not None) and (transform_file_path is not None):
        raise ValueError("Cannot pass in both transform_str"
                         " and transform_file_path.")

    if transform_str is None and transform_file_path is None:
        raise ValueError("Must pass either transform_str"
                         " or transform_file_path.")

    if transform_str is None:
        assert transform_file_path is not None
        with open(transform_file_path, "r") as fp:
            transform_str = fp.read()

    assert transform_str is not None
    transform_clbl = _get_clbl_from_string(transform_str)

    # type-ignored because last 2 arguments are optional arguments and mypy
    # cannot deduce that.
    runtime = timeit(einsum,
                     transform=transform_clbl,  # type: ignore
                     cl_ctx=cl_ctx,
                     long_dim_length=long_dim_length)

    conn = sqlite3.connect(database)

    if len(cl_ctx.devices) > 1:
        raise NotImplementedError("CL contexts with multiple devices not supported")

    cl_device, = cl_ctx.devices
    device_name = _get_cl_device_name_for_db(cl_device)
    cursor = conn.cursor()

    # {{{ get available tables

    cursor.execute(" SELECT name FROM sqlite_master"
                   f" WHERE (type='table' AND name='{device_name}');")

    if not cursor.fetchall():
        # device table not available
        logger.info(f"Table for {device_name} not in DB, creating one.")
        cursor.execute(f"CREATE TABLE {device_name} ("
                       " subscripts TEXT,"
                       " index_to_length TEXT,"
                       " use_matrix TEXT,"
                       " value_to_dtype TEXT,"
                       " loopy_transform TEXT,"
                       " runtime_in_sec REAL,"
                       " authors TEXT,"
                       " compiler_version TEXT,"
                       " cl_kernel TEXT,"
                       " giga_op_info TEXT,"
                       " timestamp TEXT,"
                       " remarks TEXT"
                       ")")

    # }}}

    subscripts = einsum.get_subscripts()
    index_to_length = _get_index_to_length_for_db(einsum)
    transform_str = transform_str.replace("\n", "\\n").replace("'", "''")
    use_matrix = _get_use_matrix_for_db(einsum).replace("\n", "\\n")
    value_to_dtype = _get_value_to_dtype_for_db(einsum)
    # type-ignored because last 2 arguments are optional arguments and mypy
    # cannot deduce that.
    cl_kernel = (lp
                 .generate_code_v2(transform_clbl(  # type: ignore
                     generate_loopy(einsum)))
                 .device_code()).replace("\n", "\\n")
    compiler_version = _get_cl_version_for_db(cl_device)
    op_info = _get_op_info_for_db(einsum, long_dim_length=long_dim_length)

    # {{{ logging values

    if log_performance_data:
        logger.info("Recorded --\n"
                    + _get_log_str_for_run(einsum,
                                           runtime=runtime,
                                           device=cl_device,
                                           long_dim_length=long_dim_length))

    # }}}

    # {{{ compute timestamp in Chicago

    import pytz
    from datetime import datetime

    timestamp = (datetime
                .now(pytz.timezone("America/Chicago")) .strftime("%Y_%m_%d_%H%M%S"))

    # }}}

    cursor.execute(f"INSERT INTO {device_name}"
                   " VALUES ("
                   f"'{subscripts}',"         # subscripts
                   f" '{index_to_length}',"   # index_to_length
                   f" '{use_matrix}',"        # use_matrix
                   f" '{value_to_dtype}',"    # value_to_dtype
                   f" '{transform_str}',"     # loopy_transform
                   f" {runtime},"             # runtime_in_sec
                   f" '{authors}',"           # authors
                   f" '{compiler_version}',"  # compiler_version
                   f" '{cl_kernel}',"         # cl_kernel
                   f" '{op_info}',"           # giga_op_info
                   f" '{timestamp}',"         # timestamp
                   f" '{remarks}'"            # remarks
                   ")")

    conn.commit()


@dataclass(frozen=True, eq=True, repr=True)
class QueryInfo:
    loopy_transform: TransformT
    runtime_in_sec: float
    authors: str
    compiler_version: str
    cl_kernel: str
    giga_op_info: PMapT[np.dtype[Any], float]
    remarks: str


def query(einsum: FusedEinsum,
          cl_ctx: "cl.Context",
          database: str = DEFAULT_TRANSFORM_ARCHIVE,
          ) -> Tuple[QueryInfo, ...]:
    """
    Returns facts of previous recorded runs of *einsum* on *cl_ctx*.
    """
    # TODO: This should  somehow solve the normalized FusedEinsum problem.
    raise NotImplementedError

# vim: foldmethod=marker