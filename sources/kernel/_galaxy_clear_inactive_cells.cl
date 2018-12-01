//
// Created by Iulian Rotaru on 2018-11-26.
//

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

inline long galaxy_down(__global galaxy_infos *infos, __global cell *cells, unsigned long cell_idx) {

    if (cell_idx >= infos->last_layer_idx) return -1;

    if (cell_idx == 0) return 1;

    return (long) ((cell_idx - cells[cell_idx].layer_idx) + pow((float) 4, (float) cells[cell_idx].depth) +
                   (cells[cell_idx].layer_idx * 4));

}

__kernel void
_galaxy_clear_inactive_cells(__global cell *cells, __global galaxy_infos *infos, __global
                             const unsigned long *start_idx) {

    const unsigned long idx = *start_idx + get_global_id(0);
    const unsigned long galaxy_idx = get_global_id(1);

    if (idx < infos[galaxy_idx].cell_count) {

        unsigned long coffset = infos[galaxy_idx].cell_buffer_offset;

        long children_cells = galaxy_down(infos + galaxy_idx, cells + coffset, idx);

        if (children_cells == -1 && cells[idx + coffset].body_count == 0) {

            cells[idx + coffset].active = 0;
            return;

        }

        if (cells[idx + coffset].body_count == 0) {

            if (!cells[children_cells + coffset].active &&
                !cells[children_cells + 1 + coffset].active &&
                !cells[children_cells + 2 + coffset].active &&
                !cells[children_cells + 3 + coffset].active) {

                cells[idx + coffset].active = 0;

            }

        }

    }

}

