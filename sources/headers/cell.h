//
// Created by mortimr on 24/11/18.
//

#ifndef OCLBHGS_CELL_H
#define OCLBHGS_CELL_H

#ifdef __APPLE__

#include <z3.h>
#include <zconf.h>

#else
#include <unistd.h>
#include <stdio.h>
#endif

#include "../kernel_headers/cell.h"
#include "./galaxy.h"


void cell_clear_idxs(ocl_galaxy *galaxy, ocl *ocl);

void cell_set_idxs(ocl_galaxy *galaxy, ocl *ocl);

void cell_set_amount(ocl_galaxy *galaxy, ocl *ocl);

#endif //OCLBHGS_CELL_H
