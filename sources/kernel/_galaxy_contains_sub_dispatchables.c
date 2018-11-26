//
// Created by Iulian Rotaru on 2018-11-26.
//

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

__kernel void
_galaxy_contains_sub_dispatchables(__global cell *cells, __global galaxy_infos *infos, __global
                                   unsigned long *found) {

    const unsigned long idx = get_global_id(0);

    if (cells[idx].body_count > 1 && cells[idx].depth < infos->depth) *found = 1;

}

