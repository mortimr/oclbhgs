//
// Created by Iulian Rotaru on 2018-11-26.
//

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

__kernel void
_body_apply_accelerations(__global body *bodies, __global cell *cells, __global galaxy_infos *infos) {

    unsigned int body_idx = get_global_id(0);
    unsigned int galaxy_idx = get_global_id(1);

    if (body_idx < infos[galaxy_idx].body_count) {

        unsigned long boffset = infos[galaxy_idx].body_buffer_offset;
        unsigned long coffset = infos[galaxy_idx].cell_buffer_offset;

        bodies[body_idx + boffset].speed.x += bodies[body_idx +
                                                     boffset].cache.x;
        bodies[body_idx + boffset].speed.y += bodies[body_idx +
                                                     boffset].cache.y;
        bodies[body_idx + boffset].pos.x += bodies[body_idx +
                                                   boffset].speed.x;
        bodies[body_idx + boffset].pos.y += bodies[body_idx +
                                                   boffset].speed.y;
        bodies[body_idx + boffset].cache.x = 0;
        bodies[body_idx + boffset].cache.y = 0;

        if ((bodies[body_idx + boffset].pos.x >
             cells[bodies[body_idx + boffset].cell_idx - 1 +
                   coffset].pos.x +
             cells[bodies[body_idx + boffset].cell_idx - 1 +
                   coffset].size.x)
            || (bodies[body_idx + boffset].pos.y >
                cells[bodies[body_idx + boffset].cell_idx - 1 +
                      coffset].pos.y +
                cells[bodies[body_idx + boffset].cell_idx - 1 +
                      coffset].size.y)
            || (bodies[body_idx + boffset].pos.x <
                cells[bodies[body_idx + boffset].cell_idx - 1 +
                      coffset].pos.x)
            || (bodies[body_idx + boffset].pos.y <
                cells[bodies[body_idx + boffset].cell_idx - 1 +
                      coffset].pos.y)) {

            bodies[body_idx + boffset].cell_idx = 0;

        }

    }
}

