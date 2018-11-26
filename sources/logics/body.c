//
// Created by mortimr on 24/11/18.
//

#ifdef __APPLE__

# include <z3.h>
# include <zconf.h>

#else

# include <unistd.h>
# include <stdio.h>

#endif

#include "../headers/body.h"

body *body_init_set(size_t amount) {

    return (body *) calloc(amount, sizeof(body));

}

void body_sort(ocl_galaxy *galaxy, ocl *ocl) {

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_BODY_SORT], 0, sizeof(cl_mem), &galaxy->bodies);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "body_sort %d: Unable to set argument 0 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_BODY_SORT], 1, sizeof(cl_mem), &galaxy->sorted_bodies);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "body_sort %d: Unable to set argument 1 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_BODY_SORT], 2, sizeof(cl_mem), &galaxy->infos);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "body_sort %d: Unable to set argument 2 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel[KERNEL_BODY_SORT], 1, NULL, &galaxy->body_count, NULL, 0,
                                      NULL, NULL);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "body_sort %d: Error while calling kernel\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clEnqueueCopyBuffer(ocl->queue, galaxy->sorted_bodies, galaxy->bodies, 0, 0,
                                   sizeof(body) * galaxy->body_count, 0, NULL, NULL);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "body_sort %d: Error copying buffers\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clFinish(ocl->queue);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "body_sort %d: Error while finishing execution queue\n", ocl->err);
        exit(ocl->err);

    }

}

void body_apply_accelerations(ocl_galaxy *galaxy, ocl *ocl) {

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_BODY_APPLY_ACCELERATIONS], 0, sizeof(cl_mem), &galaxy->bodies);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "apply_accelerations %d: Unable to set argument 0 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_BODY_APPLY_ACCELERATIONS], 1, sizeof(cl_mem), &galaxy->cells);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "apply_accelerations %d: Unable to set argument 1 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel[KERNEL_BODY_APPLY_ACCELERATIONS], 1, NULL,
                                      &galaxy->body_count, NULL, 0, NULL, NULL);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "apply_accelerations %d: Error while calling kernel\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clFinish(ocl->queue);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "apply_accelerations %d: Error while finishing execution queue\n", ocl->err);
        exit(ocl->err);

    }

}

