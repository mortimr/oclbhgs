//
// Created by Iulian Rotaru on 2018-11-26.
//

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

__kernel void
_body_sort(__global body *bodies, __global body *sorted_bodies, __global galaxy_infos *infos) {

    unsigned long idx = get_global_id(0);
    unsigned long new_pos = 0;
    body body_copy = bodies[idx];

    for (unsigned long search_idx = 0; search_idx < infos->body_count; ++search_idx) {

        if (search_idx != idx) {

            if (bodies[search_idx].cell_idx < body_copy.cell_idx) {

                ++new_pos;

            } else if (bodies[search_idx].cell_idx == body_copy.cell_idx && search_idx < idx) {

                ++new_pos;

            }

        }

    }

    sorted_bodies[new_pos] = body_copy;

}

