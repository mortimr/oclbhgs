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
_galaxy_compute_com(__global cell *cells, __global body *bodies, __global galaxy_infos *infos, __global
                    const unsigned long *start_idx) {

    const unsigned long idx = *start_idx + get_global_id(0);

    if (cells[idx].active) {

        if (cells[idx].body_count == 0) {

            long children_cells = galaxy_down(infos, cells, idx);

            if (children_cells == -1 && cells[idx].body_count == 0) {

                cells[idx].com.pos.x = 0;
                cells[idx].com.pos.y = 0;
                cells[idx].com.mass = 0;
                return;

            }

            float total_mass = 0;

            for (unsigned int children_idx = 0; children_idx < 4; ++children_idx) {

                if (cells[children_cells + children_idx].active) {

                    cells[idx].com.pos.x += cells[children_cells + children_idx].com.pos.x *
                                            cells[children_cells + children_idx].com.mass;
                    cells[idx].com.pos.y += cells[children_cells + children_idx].com.pos.y *
                                            cells[children_cells + children_idx].com.mass;
                    total_mass += cells[children_cells + children_idx].com.mass;

                }

            }

            if (total_mass) {

                cells[idx].com.pos.x /= total_mass;
                cells[idx].com.pos.y /= total_mass;
                cells[idx].com.mass = total_mass;

            }

        } else {

            float total_mass = 0;

            for (unsigned int body_idx = 0; body_idx < cells[idx].body_count; ++body_idx) {

                cells[idx].com.pos.x += bodies[cells[idx].body_idx - 1 + body_idx].pos.x *
                                        bodies[cells[idx].body_idx - 1 + body_idx].mass;
                cells[idx].com.pos.y += bodies[cells[idx].body_idx - 1 + body_idx].pos.y *
                                        bodies[cells[idx].body_idx - 1 + body_idx].mass;
                total_mass += bodies[cells[idx].body_idx - 1 + body_idx].mass;

            }

            if (total_mass) {

                cells[idx].com.pos.x /= total_mass;
                cells[idx].com.pos.y /= total_mass;
                cells[idx].com.mass = total_mass;

            }

        }

    }

}

