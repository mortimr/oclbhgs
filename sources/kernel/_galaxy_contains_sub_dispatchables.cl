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
    const unsigned long galaxy_idx = get_global_id(1);

    if (idx < infos[galaxy_idx].cell_count && found[galaxy_idx] == 0) {

        unsigned long coffset = infos[galaxy_idx].cell_buffer_offset;

        if (cells[idx + coffset].body_count > 1 &&
            cells[idx + coffset].depth < infos[galaxy_idx].depth)
            found[galaxy_idx] = 1;

    }

}

