//
// Created by Iulian Rotaru on 2018-11-26.
//

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

__kernel void
_cell_set_amount(__global body *bodies, __global cell *cells, __global galaxy_infos *infos) {

    unsigned long idx = get_global_id(0);

    if (idx == infos->body_count - 2 && bodies[idx].cell_idx > 0) {

        cells[bodies[idx + 1].cell_idx - 1].body_count =
                (idx + 1) - (cells[bodies[idx + 1].cell_idx - 1].body_idx - 1) + 1;

    }

    if (bodies[idx].cell_idx > 0 && bodies[idx + 1].cell_idx != bodies[idx].cell_idx) {

        cells[bodies[idx].cell_idx - 1].body_count = idx - (cells[bodies[idx].cell_idx - 1].body_idx - 1) + 1;

    }

}

