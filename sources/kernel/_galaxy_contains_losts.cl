//
// Created by Iulian Rotaru on 2018-11-26.
//

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

__kernel void
_galaxy_contains_losts(__global body *bodies, __global galaxy_infos *infos, __global unsigned long *found) {

    const unsigned long galaxy_idx = get_global_id(1);
    const unsigned long idx = get_global_id(0);

    if (idx < infos[galaxy_idx].body_count && found[galaxy_idx] == 0) {

        unsigned long boffset = infos[galaxy_idx].body_buffer_offset;

        if (bodies[idx + boffset].pos.x < 0 ||
            bodies[idx + boffset].pos.x > infos[galaxy_idx].map_limits.x ||
            bodies[idx + boffset].pos.y < 0 ||
            bodies[idx + boffset].pos.y > infos[galaxy_idx].map_limits.y)
            return;

        if (bodies[idx + boffset].cell_idx == 0)
            found[galaxy_idx] = 1;

    }

}

