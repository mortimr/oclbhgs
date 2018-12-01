//
// Created by Iulian Rotaru on 2018-11-26.
//

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

__kernel void
_body_sort(__global body *bodies, __global body *sorted_bodies, __global galaxy_infos *infos) {

    unsigned long galaxy_idx = get_global_id(1);
    unsigned long idx = get_global_id(0);

    if (idx < infos[galaxy_idx].body_count) {

        unsigned long boffset = infos[galaxy_idx].body_buffer_offset;

        unsigned long new_pos = 0;
        body body_copy = bodies[idx + boffset];

        for (unsigned long search_idx = 0; search_idx < infos[galaxy_idx].body_count; ++search_idx) {

            if (search_idx != idx) {

                if (bodies[search_idx + boffset].cell_idx < body_copy.cell_idx) {

                    ++new_pos;

                } else if (bodies[search_idx + boffset].cell_idx == body_copy.cell_idx && search_idx < idx) {

                    ++new_pos;

                }

            }

        }

        sorted_bodies[new_pos + boffset] = body_copy;
    }

}

