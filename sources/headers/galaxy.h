//
// Created by mortimr on 24/11/18.
//

#ifndef OCLBHGS_GALAXY_H
#define OCLBHGS_GALAXY_H

#include "../kernel_headers/galaxy.h"
#include "./ocl.h"
#include "color.h"

typedef struct ocl_galaxy {
    galaxy **galaxy;

    cl_mem cells;
    cl_mem bodies;
    cl_mem sorted_bodies;
    cl_mem infos;
    cl_mem contains_losts;
    cl_mem contains_sub_dispatchables;
    cl_mem dispatch_sub_dispatchables_start_idx;
    cl_mem clear_inactive_cells_start_idx;
    cl_mem compute_com_start_idx;
    cl_mem compute_accelerations_start_idx;
    cl_mem compute_history;

    unsigned long *depth;
    unsigned long *body_count;
    unsigned long *cell_count;
    unsigned long *max_local_work_size;
    unsigned long *last_layer_idx;
    unsigned long *body_buffer_offset;
    unsigned long *cell_buffer_offset;
    unsigned long *history_buffer_offset;

    unsigned long highest_body_count;
    unsigned long highest_cell_count;
    unsigned long highest_depth;
    unsigned long highest_depth_last_layer_index;
    unsigned long galaxy_count;
    unsigned long history_size;

    color *quadrant_color;
    color *body_color;

} ocl_galaxy;

ocl_galaxy *galaxy_allocate(unsigned int galaxy_count, ocl* ocl, unsigned int depth, unsigned int *body_counts);
void galaxy_set_colors(ocl_galaxy *galaxy, float r, float g, float b, color_target target, unsigned long galaxy_index);

void galaxy_init(ocl_galaxy *ret, ocl *ocl, cell **cells, unsigned long max_depth, float theta, float g, body *bodies,
                        unsigned long body_count, float width, float height, unsigned int galaxy_idx);

void galaxy_resolve(ocl_galaxy *galaxy, ocl *ocl);

void galaxy_recover_bodies(ocl_galaxy *galaxy, ocl *ocl, body *bodies, unsigned long galaxy_index);

void galaxy_recover_cells(ocl_galaxy *galaxy, ocl *ocl, cell *cells, unsigned long galaxy_index);

void galaxy_compute(ocl_galaxy *galaxy, ocl *ocl);

#endif //OCLBHGS_GALAXY_H
