//
// Created by Iulian Rotaru on 2018-11-26.
//

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

__kernel void
_cell_clear_idxs(__global cell *cells, __global galaxy_infos *infos) {

    unsigned int idx = get_global_id(0);
    unsigned int galaxy_idx = get_global_id(1);

    if (idx < infos[galaxy_idx].cell_count) {

        unsigned long coffset = infos[galaxy_idx].cell_buffer_offset;

        if (cells[idx + coffset].body_idx != 0) {

            cells[idx + coffset].body_idx = 0;

        }

        if (cells[idx + coffset].body_count != 0) {

            cells[idx + coffset].body_count = 0;

        }
    }

}

