//
// Created by Iulian Rotaru on 2018-11-26.
//

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

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

inline long galaxy_pos_to_cell(__global galaxy_infos *infos, __global position *pos) {

    unsigned long dec = infos->side_cell_count_lowest_level;
    unsigned long x_dec = 0;
    unsigned long y_dec = 0;
    unsigned long idx = 0;
    unsigned long x_pos;
    unsigned long y_pos;

    x_pos = (unsigned long) floor(pos->x / infos->small_cell_size.x);
    y_pos = (unsigned long) floor(pos->y / infos->small_cell_size.y);

    unsigned long counter;

    for (; dec > 1; dec /= 2) {

        counter = 0;

        if (x_pos >= (dec / 2) + x_dec) {

            ++counter;
            x_dec += dec / 2;

        }

        if (y_pos >= (dec / 2) + y_dec) {

            counter += 2;
            y_dec += dec / 2;

        }

        idx += counter * ((dec / 2) * (dec / 2));

    }

    return infos->last_layer_idx + idx;

}

__kernel void
_galaxy_dispatch_losts(__global cell *cells, __global body *bodies, __global galaxy_infos *infos) {

    const unsigned long galaxy_idx = get_global_id(1);
    const unsigned long body_idx = get_global_id(0);

    if (body_idx < infos[galaxy_idx].body_count) {

        unsigned long coffset = infos[galaxy_idx].cell_buffer_offset;
        unsigned long boffset = infos[galaxy_idx].body_buffer_offset;

        if (bodies[body_idx + boffset].pos.x < 0 ||
            bodies[body_idx + boffset].pos.x > infos[galaxy_idx].map_limits.x ||
            bodies[body_idx + boffset].pos.y < 0 ||
            bodies[body_idx + boffset].pos.y > infos[galaxy_idx].map_limits.y)
            return;

        if (bodies[body_idx + boffset].cell_idx == 0) {

            long lower_node_idx = galaxy_pos_to_cell(infos + galaxy_idx,
                                                     &bodies[body_idx + boffset].pos);
            long tmp_lower_node_idx;

            if (lower_node_idx < 0)
                return;

            while (lower_node_idx >= 0 && !cells[lower_node_idx + coffset].active) {

                if ((tmp_lower_node_idx = galaxy_up(cells + coffset, (unsigned long) lower_node_idx)) == -1) break;

                lower_node_idx = tmp_lower_node_idx;

            }


            bodies[body_idx + boffset].cell_idx = (unsigned long) lower_node_idx + 1;
            if (!cells[lower_node_idx + coffset].active)
                cells[lower_node_idx + coffset].active = 1;

        }
    }
}

