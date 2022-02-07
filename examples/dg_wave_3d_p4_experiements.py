import logging
import numpy as np
import pyopencl as cl
import feinsum as f
import loopy as lp
logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def get_grad_einsum(ndofs, ndim):
    return f.einsum("xer,rij,ej->xei",
                    f.array((ndim, np.inf, ndim,),
                            "float64"),
                    f.array((ndim, ndofs, ndofs),
                            "float64"),
                    f.array((np.inf, ndofs),
                            "float64"),
                    arg_names=["J", "R", "u"])


# 1. Tiling without putting things into shared memory.
# 2. Tiling with things in shared memory


def transform_opt_einsum_knl(t_unit):
    ncells_per_workgroup = 9
    nworkitems_per_cell = 7
    j_tile_len = 9
    i_tile_len = 35

    t_unit = lp.assignment_to_subst(t_unit, "_fe_tmp")
    t_unit = lp.split_iname(t_unit, "i_0", i_tile_len, outer_iname="i_tile")
    t_unit = lp.split_iname(t_unit, "j", j_tile_len, outer_iname="j_tile")

    t_unit = lp.rename_iname(t_unit, "i_0_inner", "i_0")
    t_unit = lp.rename_iname(t_unit, "j_inner", "j")
    t_unit = lp.rename_iname(t_unit, "x_0", "x")
    t_unit = lp.rename_iname(t_unit, "r_0", "r")
    t_unit = lp.split_iname(t_unit, "e_0", ncells_per_workgroup,
                            inner_iname="e_inner", outer_iname="e_outer",
                            outer_tag="g.0", inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, "i_0", nworkitems_per_cell,
                            inner_iname="i_inner", outer_iname="i_outer",
                            inner_tag="l.0")

    t_unit = lp.add_prefetch(t_unit,
                             "J",
                             sweep_inames=["r", "x"],
                             fetch_outer_inames=frozenset(["e_inner",
                                                           "e_outer",
                                                           "i_inner"]),
                             temporary_address_space=lp.AddressSpace.PRIVATE,
                             temporary_name="J_prftch",
                             )

    # {{{ TODO: Make precompute smarter (should be a single precompute call)

    t_unit = lp.precompute(t_unit, "_fe_tmp_subst",
                           sweep_inames=["r"],
                           precompute_outer_inames=frozenset({"e_inner",
                                                              "e_outer",
                                                              "i_inner",
                                                              "i_outer",
                                                              "i_tile",
                                                              }),
                           temporary_name="tmp_hoist",
                           temporary_address_space=lp.AddressSpace.PRIVATE,
                           compute_insn_id="insn_hoist",
                           )
    t_unit = lp.privatize_temporaries_with_inames(t_unit,
                                                  "i_outer",
                                                  only_var_names=["tmp_hoist"])

    t_unit = lp.duplicate_inames(t_unit, "i_outer", "id:insn_hoist",
                                 "i_outer_hoist")

    # }}}

    # {{{ Move 'u ' to shared.

    # I know, yuck! But, loopy cannot handle prefetching with inames belonging
    # to different domains.
    combined_domain = t_unit.default_entrypoint.combine_domains((0, 1))
    t_unit = t_unit.with_kernel(t_unit
                                .default_entrypoint
                                .copy(domains=[combined_domain]))
    if 1:
        # Prefetch 'u' within the tile
        t_unit = lp.add_prefetch(t_unit, "u",
                                 sweep_inames=["e_inner", "j"],
                                 fetch_outer_inames=frozenset(["e_outer",
                                                               "i_tile",
                                                               "j_tile"]),
                                 temporary_address_space=lp.AddressSpace.LOCAL,
                                 dim_arg_names=["e_prftch", "j_prftch"],
                                 default_tag=None,
                                 )
    elif 0:
        # Prefetch 'u' once at the top.
        t_unit = lp.add_prefetch(t_unit, "u",
                                 sweep_inames=["e_inner", "j", "j_tile"],
                                 fetch_outer_inames=frozenset(["e_outer"]),
                                 temporary_address_space=lp.AddressSpace.LOCAL,
                                 dim_arg_names=["e_prftch", "j_prftch"],
                                 default_tag=None,
                                 )
    else:
        pass

    t_unit = lp.join_inames(t_unit, ["e_prftch", "j_prftch"], "i_uprftch")
    t_unit = lp.split_iname(t_unit, "i_uprftch",
                            ncells_per_workgroup * nworkitems_per_cell,
                            outer_tag="unr")

    t_unit = lp.split_iname(t_unit, "i_uprftch_inner",
                            nworkitems_per_cell,
                            inner_tag="l.0", outer_tag="l.1")

    # }}}

    # {{{ Move 'R' to shared.

    t_unit = lp.add_prefetch(t_unit, "R",
                             sweep_inames=["r_0", "i_inner", "i_outer_hoist", "j"],
                             fetch_outer_inames=frozenset(["e_outer",
                                                           "i_tile",
                                                           "j_tile"]),
                             temporary_address_space=lp.AddressSpace.LOCAL,
                             dim_arg_names=["r_prftch", "i_prftch", "j_prftch"],
                             default_tag=None,
                             )
    t_unit = lp.join_inames(t_unit, ["r_prftch", "i_prftch", "j_prftch"],
                            "i_Rprftch")

    t_unit = lp.split_iname(t_unit, "i_Rprftch",
                            ncells_per_workgroup * nworkitems_per_cell,
                            outer_tag="unr")

    t_unit = lp.split_iname(t_unit, "i_Rprftch_inner",
                            nworkitems_per_cell,
                            inner_tag="l.0", outer_tag="l.1")

    # }}}

    # {{{ make buffer array smarter (should be a single call to buffer_array)

    t_unit = lp.buffer_array(t_unit, "_fe_out", buffer_inames=["x"],
                             init_expression="0",
                             default_tag=None, temporary_is_local=False)
    t_unit = lp.privatize_temporaries_with_inames(t_unit, "i_outer",
                                                  only_var_names={"_fe_out_buf"})

    t_unit = lp.duplicate_inames(t_unit,
                                 inames=["i_outer"],
                                 within="id:init__fe_out",
                                 new_inames=["_fe_out_init_1"])

    t_unit = lp.duplicate_inames(t_unit,
                                 inames=["i_outer"],
                                 within="id:store__fe_out",
                                 new_inames=["_fe_out_store_1"])

    # }}}

    # {{{ must be smarter way of doing this in loopy

    t_unit = lp.realize_reduction(t_unit, insn_id_filter="insn_hoist")
    t_unit = lp.privatize_temporaries_with_inames(t_unit,
                                                  frozenset(["r_0",
                                                             "i_outer_hoist"]),
                                                  only_var_names={"acc_j_tile_j"})
    t_unit = lp.duplicate_inames(t_unit,
                                 ["i_outer_hoist", "r_0"],
                                 within="id:insn_hoist_j_tile_j_init",
                                 new_inames=["i_outer_hoist_init", "r_0_init"],
                                 )

    t_unit = lp.duplicate_inames(t_unit,
                                 ["i_outer_hoist", "r_0"],
                                 within="id:insn_hoist",
                                 new_inames=["i_outer_hoist_store", "r_0_store"],
                                 )

    # }}}

    # t_unit = lp.tag_inames(t_unit, "x:unr, r:unr")
    # t_unit = lp.set_options(t_unit, "write_cl")

    return t_unit


def main():
    from feinsum.data.device_info import DEV_TO_PEAK_GFLOPS
    cl_ctx = cl.create_some_context()
    logger.info(f"Running on <{cl_ctx}>")

    expr = get_grad_einsum(ndofs=35, ndim=3)

    if len(cl_ctx.devices) != 1:
        logger.info("Multiple devices in the context")
        return
    if cl_ctx.devices[0].name not in DEV_TO_PEAK_GFLOPS:
        logger.info("Device not known.")
        return

    print(
        f.stringify_comparison_vs_roofline(
            expr,
            cl_ctx=cl_ctx,
            transform=transform_opt_einsum_knl,
            schedule=f.get_opt_einsum_contraction_schedule(expr)))


if __name__ == "__main__":
    main()
