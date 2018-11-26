//
// Created by mortimr on 24/11/18.
//

#ifndef OCLBHGS_KERNEL_BODY_H
#define OCLBHGS_KERNEL_BODY_H

#include "position.h"
#include "mass.h"
#include "cell.h"

typedef struct body {
    position pos;
    position cache;
    position speed;
    mass mass;
    unsigned long cell_idx;
} body;

#endif //OCLBHGS_KERNEL_BODY_H
