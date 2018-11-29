//
// Created by Iulian Rotaru on 2018-11-26.
//

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

__kernel void
_cell_set_idxs(__global body *bodies, __global cell *cells) {

    unsigned long idx = get_global_id(0);

    if (idx == 0 && bodies[idx].cell_idx != 0) {

        cells[bodies[idx].cell_idx - 1].body_idx = idx + 1;

    }

    if (bodies[idx + 1].cell_idx != bodies[idx].cell_idx) {

        cells[bodies[idx + 1].cell_idx - 1].body_idx = idx + 2;

    }

}

