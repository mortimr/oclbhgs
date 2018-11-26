//
// Created by mortimr on 24/11/18.
//

#ifndef OCLBHGS_KERNEL_CELL_H
#define OCLBHGS_KERNEL_CELL_H

#include "com.h"

typedef struct cell {
    char active;
    unsigned long depth;
    com com;
    unsigned long body_idx;
    unsigned long body_count;
    position pos;
    position size;
    unsigned long layer_idx;
} cell;

#endif //OCLBHGS_KERNEL_CELL_H
