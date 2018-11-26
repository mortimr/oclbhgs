//
// Created by Iulian Rotaru on 2018-11-28.
//

#include "../headers/cell.h"

void cell_clear_idxs(ocl_galaxy *galaxy, ocl *ocl) {

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_CELL_CLEAR_IDXS], 0, sizeof(cl_mem), &galaxy->cells);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "clear_idxs %d: Unable to set argument 0 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel[KERNEL_CELL_CLEAR_IDXS], 1, NULL, &galaxy->cell_count,
                                      NULL, 0, NULL, NULL);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "clear_idxs %d: Error while calling kernel\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clFinish(ocl->queue);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "clear_idxs %d: Error while finishing execution queue\n", ocl->err);
        exit(ocl->err);

    }

}

void cell_set_idxs(ocl_galaxy *galaxy, ocl *ocl) {

    unsigned long body_count = galaxy->body_count - 1;

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_CELL_SET_IDXS], 0, sizeof(cl_mem), &galaxy->bodies);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "set_idxs %d: Unable to set argument 0 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_CELL_SET_IDXS], 1, sizeof(cl_mem), &galaxy->cells);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "set_idxs %d: Unable to set argument 1 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel[KERNEL_CELL_SET_IDXS], 1, NULL, &body_count, NULL, 0,
                                      NULL, NULL);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "set_idxs %d: Error while calling kernel\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clFinish(ocl->queue);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "set_idxs %d: Error while finishing execution queue\n", ocl->err);
        exit(ocl->err);

    }

}

void cell_set_amount(ocl_galaxy *galaxy, ocl *ocl) {

    unsigned long body_count = galaxy->body_count - 1;

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_CELL_SET_AMOUNT], 0, sizeof(cl_mem), &galaxy->bodies);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "set_amount %d: Unable to set argument 0 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_CELL_SET_AMOUNT], 1, sizeof(cl_mem), &galaxy->cells);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "set_amount %d: Unable to set argument 1 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_CELL_SET_AMOUNT], 2, sizeof(cl_mem), &galaxy->infos);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "set_amount %d: Unable to set argument 2 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel[KERNEL_CELL_SET_AMOUNT], 1, NULL, &body_count, NULL, 0,
                                      NULL, NULL);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "set_amount %d: Error while calling kernel\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clFinish(ocl->queue);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "set_amount %d: Error while finishing execution queue\n", ocl->err);
        exit(ocl->err);

    }

}

