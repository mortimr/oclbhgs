//
// Created by mortimr on 25/11/18.
//

#ifndef OCLBHGS_KERNEL_DEV_H
#define OCLBHGS_KERNEL_DEV_H

#define __kernel
#define __global
#define __local

#define get_global_id
#define get_local_size
#define get_group_id
#define barrier(e) (0)

#define CLK_LOCAL_MEM_FENCE
#define CLK_GLOBAL_MEM_FENCE

#define floor
#define pow
#define sqrt
#define isnan

#define atom_add
#define atom_sub

#endif //OCLBHGS_KERNEL_DEV_H
