//
// Created by Iulian Rotaru on 2018-11-26.
//

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

__kernel void
_cell_clear_idxs(__global cell *cells) {

    unsigned int idx = get_global_id(0);

    if (cells[idx].body_idx != 0) {

        cells[idx].body_idx = 0;

    }

    if (cells[idx].body_count != 0) {

        cells[idx].body_count = 0;

    }

}

