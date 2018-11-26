//
// Created by mortimr on 25/11/18.
//

#ifndef OCLBHGS_KERNEL_OCL_H
#define OCLBHGS_KERNEL_OCL_H

#ifdef __APPLE__

#include <OpenCL/cl.h>

#else
#include <CL/cl_platform.h>
#include <CL/cl.h>
#endif

#define SOURCE_FILE_COUNT (12)

typedef enum kernels {
    KERNEL_GALAXY_CONTAINS_LOSTS = 0,
    KERNEL_GALAXY_DISPATCH_LOSTS,
    KERNEL_GALAXY_CONTAINS_SUB_DISPATCHABLES,
    KERNEL_GALAXY_DISPATCH_SUB_DISPATCHABLES,
    KERNEL_GALAXY_CLEAR_INACTIVE_CELLS,
    KERNEL_GALAXY_COMPUTE_COM,
    KERNEL_GALAXY_COMPUTE_ACCELERATIONS,
    KERNEL_BODY_SORT,
    KERNEL_BODY_APPLY_ACCELERATIONS,
    KERNEL_CELL_CLEAR_IDXS,
    KERNEL_CELL_SET_IDXS,
    KERNEL_CELL_SET_AMOUNT
} kernels;

typedef struct ocl {
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_program program[SOURCE_FILE_COUNT];
    cl_command_queue queue;
    cl_event event;
    cl_kernel kernel[SOURCE_FILE_COUNT];
} ocl;

#endif //OCLBHGS_KERNEL_OCL_H
