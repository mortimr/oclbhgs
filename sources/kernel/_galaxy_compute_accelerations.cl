//
// Created by Iulian Rotaru on 2018-11-26.
//

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

inline long galaxy_up(__global cell *cells, unsigned long cell_idx) {

    if (cell_idx == 0) return -1;

    if (cell_idx < 5) return 0;

    unsigned long cell_idx_count = 5;

    for (unsigned long idx = 2; idx < cells[cell_idx].depth; ++idx) {

        cell_idx_count += pow((float) 4, (float) idx);

    }

    return (long) ((cell_idx_count - pow((float) 4, (float) (cells[cell_idx].depth - 1)) +
                    ((cells[cell_idx].layer_idx) / 4)));

}

__kernel void
_galaxy_compute_accelerations(__global cell *cells, __global body *bodies, __global galaxy_infos *infos, __global
                              unsigned int *compute_history, __global const unsigned long *start_idx) {

    const unsigned long body_idx = get_global_id(0);
    const unsigned long cell_idx = *start_idx + get_global_id(1);

    if (cells[cell_idx].active) {

        long up_node = galaxy_up(cells, cell_idx);
        if (up_node >= 0 && compute_history[body_idx * infos->cell_count + up_node] != 0) {

            compute_history[body_idx * infos->cell_count + cell_idx] = 1;
            return;

        }

        float dist = sqrt(pow((float) (bodies[body_idx].pos.x - cells[cell_idx].com.pos.x), (float) 2) +
                             pow((float) (bodies[body_idx].pos.y - cells[cell_idx].com.pos.y), (float) 2));

        if (cells[cell_idx].body_count || (cells[cell_idx].size.x / dist < infos->theta)) {

            float k = infos->g * cells[cell_idx].com.mass / (pow((float) (dist + 3), (float) 3));
            float x_dir = k * (cells[cell_idx].com.pos.x - bodies[body_idx].pos.x);
            float y_dir = k * (cells[cell_idx].com.pos.y - bodies[body_idx].pos.y);

            if (!isnan(x_dir))
                bodies[body_idx].cache.x += x_dir;

            if (!isnan(y_dir))
                bodies[body_idx].cache.y += y_dir;

            compute_history[body_idx * infos->cell_count + cell_idx] = 1;

        }

    }

}

