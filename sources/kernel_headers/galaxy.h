//
// Created by mortimr on 24/11/18.
//

#ifndef OCLBHGS_KERNEL_GALAXY_H
#define OCLBHGS_KERNEL_GALAXY_H

#include "body.h"
#include "cell.h"

typedef struct galaxy_infos {
    unsigned long cell_count;
    unsigned long body_count;
    unsigned long depth;
    position map_limits;
    position small_cell_size;
    unsigned long side_cell_count_lowest_level;
    float theta;
    float g;
    unsigned long max_local_work_size;
    unsigned long last_layer_idx;
} galaxy_infos;

typedef struct galaxy {
    cell *cells;
    body *bodies;
    galaxy_infos *infos;
} galaxy;

#endif //OCLBHGS_KERNEL_GALAXY_H
