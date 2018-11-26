//
// Created by mortimr on 24/11/18.
//

#ifndef OCLBHGS_GALAXY_H
#define OCLBHGS_GALAXY_H

#include "../kernel_headers/galaxy.h"
#include "./ocl.h"

typedef struct ocl_galaxy {
    galaxy *galaxy;
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
    unsigned long depth;
    unsigned long body_count;
    unsigned long cell_count;
    unsigned long max_local_work_size;
    unsigned long last_layer_idx;
} ocl_galaxy;

ocl_galaxy *galaxy_init(ocl *ocl, cell **cells, unsigned long max_depth, float theta, float g, body *bodies,
                        unsigned long body_count, float width, float height);

void galaxy_resolve(ocl_galaxy *galaxy, ocl *ocl);

void galaxy_recover_bodies(ocl_galaxy *galaxy, ocl *ocl, body *bodies);

void galaxy_recover_cells(ocl_galaxy *galaxy, ocl *ocl, cell *cells);

void galaxy_compute(ocl_galaxy *galaxy, ocl *ocl);

#endif //OCLBHGS_GALAXY_H
