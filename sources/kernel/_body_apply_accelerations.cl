//
// Created by Iulian Rotaru on 2018-11-26.
//

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

__kernel void
_body_apply_accelerations(__global body *bodies, __global cell *cells) {

    unsigned int body_idx = get_global_id(0);

    bodies[body_idx].speed.x += bodies[body_idx].cache.x;
    bodies[body_idx].speed.y += bodies[body_idx].cache.y;
    bodies[body_idx].pos.x += bodies[body_idx].speed.x;
    bodies[body_idx].pos.y += bodies[body_idx].speed.y;
    bodies[body_idx].cache.x = 0;
    bodies[body_idx].cache.y = 0;

    if ((bodies[body_idx].pos.x > cells[bodies[body_idx].cell_idx - 1].pos.x + cells[bodies[body_idx].cell_idx - 1].size.x)
        || (bodies[body_idx].pos.y > cells[bodies[body_idx].cell_idx - 1].pos.y + cells[bodies[body_idx].cell_idx - 1].size.y)
        || (bodies[body_idx].pos.x < cells[bodies[body_idx].cell_idx - 1].pos.x)
        || (bodies[body_idx].pos.y < cells[bodies[body_idx].cell_idx - 1].pos.y)) {

        bodies[body_idx].cell_idx = 0;

    }
}

