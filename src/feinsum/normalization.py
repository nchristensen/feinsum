"""
.. autofunction:: normalize_einsum
"""

from typing import List, Dict
#from pyrsistent import pmap
from immutabledict import immutabledict as pmap
from feinsum.einsum import (FusedEinsum, SizeParam, FreeAxis, SummationAxis,
                            EinsumAxisAccess)


def normalize_einsum(einsum: FusedEinsum) -> FusedEinsum:
    """
    Returns a normalized form of *einsum*.
    """
    nfree_indices = einsum.ndim
    nredn_indices = len([idx
                         for idx in einsum.index_to_dim_length()
                         if isinstance(idx, SummationAxis)])

    # there are only 26 letters :)
    assert nfree_indices + nredn_indices <= 26

    index_to_new_name = {}
    # type-ignore reason: List is invariant
    sorted_axes: List[EinsumAxisAccess] = ([FreeAxis(i)  # type: ignore[assignment]
                                            for i in range(nfree_indices)]
                                           + [SummationAxis(i)  # type: ignore[misc]
                                              for i in range(nredn_indices)])
    for idx, ichr in zip(sorted_axes, range(97, 123)):
        index_to_new_name[idx] = chr(ichr)

    old_value_to_new_value: Dict[str, str] = {}
    new_use_matrix = []
    for use_row in einsum.use_matrix:
        new_use_row = []
        for values in use_row:
            if len(values) > 1:
                raise NotImplementedError("Multi-values per use not yet supported.")
            old_value, = values
            new_value = old_value_to_new_value.setdefault(
                old_value,
                f"arg_{len(old_value_to_new_value)}")
            new_use_row.append(frozenset([new_value]))

        new_use_matrix.append(tuple(new_use_row))

    new_value_to_dtypes = {old_value_to_new_value[old_val]: dtype
                           for old_val, dtype in sorted(einsum.value_to_dtype.items(), key=lambda e: e[0])}

               
    old_size_param_to_new_size_param: Dict[SizeParam, SizeParam] = {
        old_sz_par: SizeParam(f"N_{index_to_new_name[old_idx]}")
        for old_idx, old_sz_par in einsum.index_to_dim_length().items()
        if isinstance(old_sz_par, SizeParam)
    }

    # If we have multiple free indices and all of those are integers, then replace the longest value with
    # a SizeParam
    #"""
    non_integer_free_axes = [(old_idx, old_sz_par) for old_idx, old_sz_par in einsum.index_to_dim_length().items() if isinstance(old_idx, FreeAxis) and not isinstance(old_sz_par, int)]
    if len(non_integer_free_axes) == 0:
        integer_free_axes = sorted([(old_idx, old_sz_par) for old_idx, old_sz_par in einsum.index_to_dim_length().items() if isinstance(old_idx, FreeAxis)], key=lambda e: e[1], reverse=True)

        if len(integer_free_axes) > 1 and integer_free_axes[0][1] != integer_free_axes[1][1]:
            old_idx, old_sz_par = integer_free_axes[0]
            old_size_param_to_new_size_param[old_sz_par] = SizeParam(f"N_{index_to_new_name[old_idx]}")
    #"""     

    #print(einsum.index_to_dim_length())
    #exit()
    """
    shape_set = set()
    for arg_shape in einsum.arg_shapes:
        for entry in arg_shape:
            shape_set |= {entry}

    #print("SHAPE_SET", shape_set)
    from feinsum import VeryLongAxis
    shape_set = list(shape_set)
    dim_map = dict(zip(shape_set, shape_set))
    max_dim = None
    if len(shape_set) > 1 and all([isinstance(entry, int) for entry in shape_set]):
        max_dim = max(shape_set)
    #if max_dim is not None:
        

        dim_map[max_dim] = VeryLongAxis()
    """
    # type-ignore reason: mypy isn't smart to see that only SizeParams of the
    # dict are queried
    new_arg_shapes = tuple(
        tuple(old_size_param_to_new_size_param[dim]
              if dim in old_size_param_to_new_size_param#isinstance(dim, SizeParam)
              else dim#dim_map[dim]#dim
              for dim in old_arg_shape)
        for old_arg_shape in einsum.arg_shapes
    )

   

    #print("HERE I AM")

    #print(shape_set)
    #print(einsum.access_descriptors)
    #print(einsum)
    #print(new_arg_shapes)
    #print(einsum.index)

    new_einsum = FusedEinsum(new_arg_shapes,
                       pmap(new_value_to_dtypes),
                       einsum.access_descriptors,
                       tuple(new_use_matrix),
                       pmap(index_to_new_name))
    return new_einsum
