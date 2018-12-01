//
// Created by Iulian Rotaru on 2018-11-26.
//

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#ifdef OPENCL_HOST

#include "../sources/kernel_headers/kernel_dev.h"

#endif

#include "../sources/kernel_headers/galaxy.h"

inline long galaxy_down(__global galaxy_infos *infos, __global cell *cells, unsigned long cell_idx) {

    if (cell_idx > infos->cell_count - pow((float) 4, (float) infos->depth)) return -1;

    if (cell_idx == 0) return 1;

    return (long) ((cell_idx - cells[cell_idx].layer_idx) + pow((float) 4, (float) cells[cell_idx].depth) +
                   (cells[cell_idx].layer_idx * 4));

}

__kernel void
_galaxy_dispatch_sub_dispatchables(__global cell *cells, __global body *bodies, __global galaxy_infos *infos, __global
                                   const unsigned long *start_idx) {

    unsigned long galaxy_idx = get_global_id(1);
    unsigned long current = *start_idx + get_global_id(0);

    if (current < infos[galaxy_idx].cell_count) {

        unsigned long coffset = infos[galaxy_idx].cell_buffer_offset;
        unsigned long boffset = infos[galaxy_idx].body_buffer_offset;

        if (cells[current + coffset].active) {

            if (cells[current + coffset].body_count > 1) {

                unsigned long children_cells_idx = (unsigned long) galaxy_down(infos + galaxy_idx, cells + coffset, current);
                unsigned long chosen_children_cell_idx;

                for (unsigned long body_idx = 0; body_idx < cells[current + coffset].body_count; ++body_idx) {

                    chosen_children_cell_idx = 0;
                    if ((bodies[body_idx + cells[current + coffset].body_idx - 1 + boffset].pos.x -
                         cells[current + coffset].pos.x) / (cells[current + coffset].size.x / 2.0) >= 1.0)
                        ++chosen_children_cell_idx;

                    if ((bodies[body_idx + cells[current + coffset].body_idx - 1 + boffset].pos.y -
                         cells[current + coffset].pos.y) / (cells[current + coffset].size.y / 2.0) >= 1.0)
                        chosen_children_cell_idx += 2;

                    bodies[body_idx + cells[current + coffset].body_idx - 1 + boffset].cell_idx =
                            (children_cells_idx + chosen_children_cell_idx) + 1;

                    if (!cells[children_cells_idx + chosen_children_cell_idx + coffset].active)
                        cells[children_cells_idx + chosen_children_cell_idx + coffset].active = 1;

                }

            }

        }
    }

}

