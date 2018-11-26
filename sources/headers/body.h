//
// Created by mortimr on 24/11/18.
//

#ifndef OCLBHGS_BODY_H
#define OCLBHGS_BODY_H

#include <glob.h>

#include "../kernel_headers/body.h"
#include "galaxy.h"

void body_sort_set(body *bodies, size_t body_count);

long body_set_cell_idxs(body *bodies, unsigned long body_count, cell *cells, unsigned long cell_count);

body *body_init_set(size_t amount);

void body_sort(ocl_galaxy *galaxy, ocl *ocl);

void body_apply_accelerations(ocl_galaxy *galaxy, ocl *ocl);


#endif //OCLBHGS_BODY_H
