//
// Created by Iulian Rotaru on 2018-11-26.
//

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

// TODO check first case

__kernel void
_cell_set_amount(__global body *bodies, __global cell *cells, __global galaxy_infos *infos) {

    unsigned long idx = get_global_id(0);
    unsigned long galaxy_idx = get_global_id(1);

    if (idx < infos[galaxy_idx].body_count - 1) {

        unsigned long coffset = infos[galaxy_idx].cell_buffer_offset;
        unsigned long boffset = infos[galaxy_idx].body_buffer_offset;

        if (idx == infos[galaxy_idx].body_count - 2 &&
            bodies[idx + boffset].cell_idx > 0) {

            cells[bodies[idx + 1 + boffset].cell_idx - 1 +
                  coffset].body_count =
                    (idx + 1) - (cells[bodies[idx + 1 + boffset].cell_idx - 1 +
                                       coffset].body_idx - 1) + 1;

        }

        if (bodies[idx + boffset].cell_idx > 0 &&
            bodies[idx + 1 + boffset].cell_idx !=
            bodies[idx + boffset].cell_idx) {

            cells[bodies[idx + boffset].cell_idx - 1 +
                  coffset].body_count =
                    idx - (cells[bodies[idx + boffset].cell_idx - 1 +
                                 coffset].body_idx - 1) + 1;

        }

    }

}

