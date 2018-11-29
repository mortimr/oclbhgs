//
// Created by mortimr on 25/11/18.
//

#ifdef __APPLE__

# include <zconf.h>

#else

# include <unistd.h>

#endif

#include <printf.h>
#include <stdio.h>
#include <memory.h>
#include "../headers/ocl.h"

#ifndef PATH_TO_KERNEL

# define PATH_TO_KERNEL "../sources/kernel"

#endif

#define _KERNEL_NAME(PATH, NAME) PATH#NAME
#define KERNEL_NAME(PATH, NAME) _KERNEL_NAME(PATH, NAME)

const char *kernel_source_files[] = {
        KERNEL_NAME(PATH_TO_KERNEL, /_galaxy_contains_losts.cl),
        KERNEL_NAME(PATH_TO_KERNEL, /_galaxy_dispatch_losts.cl),
        KERNEL_NAME(PATH_TO_KERNEL, /_galaxy_contains_sub_dispatchables.cl),
        KERNEL_NAME(PATH_TO_KERNEL, /_galaxy_dispatch_sub_dispatchables.cl),
        KERNEL_NAME(PATH_TO_KERNEL, /_galaxy_clear_inactive_cells.cl),
        KERNEL_NAME(PATH_TO_KERNEL, /_galaxy_compute_com.cl),
        KERNEL_NAME(PATH_TO_KERNEL, /_galaxy_compute_accelerations.cl),
        KERNEL_NAME(PATH_TO_KERNEL, /_body_sort.cl),
        KERNEL_NAME(PATH_TO_KERNEL, /_body_apply_accelerations.cl),
        KERNEL_NAME(PATH_TO_KERNEL, /_cell_clear_idxs.cl),
        KERNEL_NAME(PATH_TO_KERNEL, /_cell_set_idxs.cl),
        KERNEL_NAME(PATH_TO_KERNEL, /_cell_set_amount.cl)
};

const char *kernel_names[] = {
        "_galaxy_contains_losts",
        "_galaxy_dispatch_losts",
        "_galaxy_contains_sub_dispatchables",
        "_galaxy_dispatch_sub_dispatchables",
        "_galaxy_clear_inactive_cells",
        "_galaxy_compute_com",
        "_galaxy_compute_accelerations",
        "_body_sort",
        "_body_apply_accelerations",
        "_cell_clear_idxs",
        "_cell_set_idxs",
        "_cell_set_amount"
};

void ocl_init(ocl *ocl) {

    cl_uint platforms, devices;

    ocl->err = clGetPlatformIDs(1, &ocl->platform, &platforms);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "ocl: Cannot get platform IDs\n");
        exit(ocl->err);

    }

    ocl->err = clGetDeviceIDs(ocl->platform, CL_DEVICE_TYPE_GPU, 1, &ocl->device, &devices);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "ocl: Cannot get device IDs\n");
        exit(ocl->err);

    }

    cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties) ocl->platform,
            0
    };

    ocl->ctx = clCreateContext(properties, 1, &ocl->device, NULL, NULL, &ocl->err);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "ocl: Cannot create OpenCL context\n");
        exit(ocl->err);

    }

    ocl->queue = clCreateCommandQueue(ocl->ctx, ocl->device, 0, &ocl->err);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "ocl: Cannot create command queue\n");
        exit(ocl->err);

    }

    for (unsigned long kernel_source_idx = 0; kernel_source_idx < SOURCE_FILE_COUNT; ++kernel_source_idx) {

        char *source_code;
        FILE *source_file;

        if (!(source_file = fopen(kernel_source_files[kernel_source_idx], "r"))) {

            dprintf(STDERR_FILENO, "ocl: Cannot read kernel source file %s \n", kernel_source_files[kernel_source_idx]);
            exit(1);

        }

        fseek(source_file, 0, SEEK_END);
        size_t source_file_size = (size_t) ftell(source_file);
        fseek(source_file, 0, SEEK_SET);

        if (!(source_code = malloc((source_file_size + 1) * sizeof(char)))) {

            dprintf(STDERR_FILENO, "ocl: Cannot allocate memory to store source for kernel %s \n",
                    kernel_source_files[kernel_source_idx]);
            exit(1);

        }

        if (fread(source_code, source_file_size, 1, source_file) < 0) {

            dprintf(STDERR_FILENO, "ocl: Cannot read kernel source file %s \n", kernel_source_files[kernel_source_idx]);
            exit(1);

        }

        fclose(source_file);

        source_code[source_file_size] = 0;

        ocl->program[kernel_source_idx] = clCreateProgramWithSource(ocl->ctx,
                                                                    1, (const char **) &source_code, &source_file_size,
                                                                    &ocl->err);
        if (ocl->err) {

            dprintf(STDERR_FILENO, "ocl: Cannot create program with source %s\n",
                    kernel_source_files[kernel_source_idx]);
            exit(ocl->err);

        }

        ocl->err = clBuildProgram(ocl->program[kernel_source_idx], 0, NULL, "", NULL, NULL);

        if (ocl->err) {

            dprintf(STDERR_FILENO, "ocl: Cannot build program with source %s\n",
                    kernel_source_files[kernel_source_idx]);

            size_t log_size;
            clGetProgramBuildInfo(ocl->program[kernel_source_idx], ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                                  &log_size);

            char *log = (char *) malloc(log_size + 1);
            log[log_size] = 0;
            clGetProgramBuildInfo(ocl->program[kernel_source_idx], ocl->device, CL_PROGRAM_BUILD_LOG, log_size, log,
                                  NULL);
            dprintf(STDERR_FILENO, "%s\n", log);
            exit(ocl->err);

        }

        ocl->kernel[kernel_source_idx] = clCreateKernel(ocl->program[kernel_source_idx],
                                                        kernel_names[kernel_source_idx], &ocl->err);

        if (ocl->err) {

            dprintf(STDERR_FILENO, "ocl: Cannot create kernel with source %s\n",
                    kernel_source_files[kernel_source_idx]);
            exit(ocl->err);

        }

    }

}