//
// Created by Iulian Rotaru on 2018-11-26.
//

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

__kernel void
_galaxy_contains_losts(__global body *bodies, __global galaxy_infos *infos, __global unsigned long *found) {

    if (*found == 0) {

        const unsigned long idx = get_global_id(0);

        if (bodies[idx].pos.x < 0 || bodies[idx].pos.x > infos->map_limits.x || bodies[idx].pos.y < 0 ||
            bodies[idx].pos.y > infos->map_limits.y)
            return;

        if (bodies[idx].cell_idx == 0)
            *found = 1;

    }

}

