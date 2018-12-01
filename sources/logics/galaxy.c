//
// Created by mortimr on 24/11/18.
//

#ifdef __APPLE__

#else

# include <unistd.h>
# include <stdio.h>
# include <stdbool.h>

#endif

#include <math.h>
#include "../headers/body.h"
#include "../headers/cell.h"

void galaxy_set_colors(ocl_galaxy *galaxy, float r, float g, float b, color_target target, unsigned long galaxy_index) {
    switch (target) {
        case BODY:
            galaxy->body_color[galaxy_index].r = r;
            galaxy->body_color[galaxy_index].g = g;
            galaxy->body_color[galaxy_index].b = b;
            break;
        case QUADRANT:
            galaxy->quadrant_color[galaxy_index].r = r;
            galaxy->quadrant_color[galaxy_index].g = g;
            galaxy->quadrant_color[galaxy_index].b = b;
            break;
    }
}

ocl_galaxy *galaxy_allocate(unsigned int galaxy_count, ocl *ocl, unsigned int depth, unsigned int *body_counts) {

    ocl_galaxy *ret;
    unsigned int cell_counts[galaxy_count];

    for (size_t idx = 0; idx < galaxy_count; ++idx) {
        cell_counts[idx] = 0;

        for (unsigned long sub_idx = 0; sub_idx <= depth; ++sub_idx) {

            cell_counts[idx] += (unsigned long) pow(4, sub_idx);

        }

    }


    if (!(ret = calloc(1, sizeof(ocl_galaxy))))
        return NULL;

    // TODO check ret
    ret->galaxy = calloc(galaxy_count, sizeof(galaxy *));
    ret->depth = calloc(galaxy_count, sizeof(unsigned long));
    ret->body_count = calloc(galaxy_count, sizeof(unsigned long));
    ret->cell_count = calloc(galaxy_count, sizeof(unsigned long));
    ret->max_local_work_size = calloc(galaxy_count, sizeof(unsigned long));
    ret->last_layer_idx = calloc(galaxy_count, sizeof(unsigned long));
    ret->body_buffer_offset = calloc(galaxy_count, sizeof(unsigned long));
    ret->cell_buffer_offset = calloc(galaxy_count, sizeof(unsigned long));
    ret->history_buffer_offset = calloc(galaxy_count, sizeof(unsigned long));

    ret->quadrant_color = calloc(galaxy_count, sizeof(color));
    ret->body_color = calloc(galaxy_count, sizeof(color));

    size_t total_cell_count = 0;
    for (size_t idx = 0; idx < galaxy_count; ++idx) {
        ret->cell_buffer_offset[idx] = total_cell_count;
        total_cell_count += cell_counts[idx];
    }

    size_t total_body_count = 0;
    for (size_t idx = 0; idx < galaxy_count; ++idx) {
        ret->body_buffer_offset[idx] = total_body_count;
        total_body_count += body_counts[idx];
    }

    ret->cells = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE, sizeof(cell) * total_cell_count, NULL, &ocl->err);
    ret->bodies = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE, sizeof(body) * total_body_count, NULL, &ocl->err);
    ret->sorted_bodies = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE, sizeof(body) * total_body_count, NULL, &ocl->err);
    ret->infos = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE, sizeof(galaxy_infos) * galaxy_count, NULL, &ocl->err);
    ret->contains_losts = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE, sizeof(unsigned int) * galaxy_count, NULL,
                                         &ocl->err);
    ret->contains_sub_dispatchables = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE, sizeof(unsigned int) * galaxy_count,
                                                     NULL,
                                                     &ocl->err);
    ret->dispatch_sub_dispatchables_start_idx = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE,
                                                               sizeof(unsigned int) * galaxy_count, NULL,
                                                               &ocl->err);
    ret->clear_inactive_cells_start_idx = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE,
                                                         sizeof(unsigned int) * galaxy_count, NULL,
                                                         &ocl->err);
    ret->compute_com_start_idx = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE, sizeof(unsigned int) * galaxy_count, NULL,
                                                &ocl->err);
    ret->compute_accelerations_start_idx = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL,
                                                          &ocl->err);

    ret->highest_body_count = 0;
    ret->highest_cell_count = 0;
    ret->highest_depth = 0;

    ret->galaxy_count = galaxy_count;

    size_t total_history_size = 0;
    for (size_t idx = 0; idx < galaxy_count; ++idx) {
        ret->history_buffer_offset[idx] = total_history_size;
        total_history_size += body_counts[idx] * cell_counts[idx];
    }

    ret->history_size = total_history_size;

    ret->compute_history = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE, sizeof(unsigned int) * total_history_size,
                                          NULL, &ocl->err);

    return ret;

}

void galaxy_init(ocl_galaxy *ret, ocl *ocl, cell **cells, unsigned long max_depth, float theta, float g, body *bodies,
                 unsigned long body_count, float width, float height, unsigned int galaxy_index) {

    unsigned long cell_count = 0;

    for (unsigned long idx = 0; idx <= max_depth; ++idx) {

        cell_count += (unsigned long) pow(4, idx);

    }

    if (!(ret->galaxy[galaxy_index] = (galaxy *) calloc(1, sizeof(galaxy))))
        return;

    if (!(ret->galaxy[galaxy_index]->cells = (cell *) calloc(cell_count, sizeof(cell))))
        return;

    if (!(ret->galaxy[galaxy_index]->infos = (galaxy_infos *) calloc(1, sizeof(galaxy_infos))))
        return;

    unsigned long cell_idx = 1;
    position cell_size = {width / 2, height / 2};

    ret->galaxy[galaxy_index]->cells[0].pos.x = 0;
    ret->galaxy[galaxy_index]->cells[0].pos.y = 0;
    ret->galaxy[galaxy_index]->cells[0].size.x = width;
    ret->galaxy[galaxy_index]->cells[0].size.y = height;
    for (unsigned long idx = 1; idx <= max_depth; ++idx) {

        unsigned long previous_level_pow = (unsigned long) pow(4, idx - 1);
        unsigned long level_pow = (unsigned long) pow(4, idx);
        for (unsigned long quad_idx = 0; quad_idx < level_pow; ++quad_idx) {

            ret->galaxy[galaxy_index]->cells[cell_idx + quad_idx].depth = idx;
            ret->galaxy[galaxy_index]->cells[cell_idx + quad_idx].size.x = cell_size.x;
            ret->galaxy[galaxy_index]->cells[cell_idx + quad_idx].size.y = cell_size.y;
            ret->galaxy[galaxy_index]->cells[cell_idx + quad_idx].layer_idx = quad_idx;

            unsigned long parent_layer_idx = (unsigned long) (cell_idx - previous_level_pow + floor(quad_idx / 4));

            ret->galaxy[galaxy_index]->cells[cell_idx + quad_idx].pos.x =
                    ret->galaxy[galaxy_index]->cells[parent_layer_idx].pos.x +
                    (quad_idx % 2) * ret->galaxy[galaxy_index]->cells[cell_idx + quad_idx].size.x;
            ret->galaxy[galaxy_index]->cells[cell_idx + quad_idx].pos.y = (float) (
                    ret->galaxy[galaxy_index]->cells[parent_layer_idx].pos.y +
                    floor((quad_idx % 4) / 2) *
                    ret->galaxy[galaxy_index]->cells[cell_idx + quad_idx].size.y);

        }

        cell_idx += level_pow;
        cell_size.x /= 2;
        cell_size.y /= 2;

    }

    ret->galaxy[galaxy_index]->infos->cell_count = cell_count;
    ret->galaxy[galaxy_index]->infos->depth = max_depth;
    ret->galaxy[galaxy_index]->infos->g = g;
    ret->galaxy[galaxy_index]->infos->theta = theta;
    ret->galaxy[galaxy_index]->bodies = bodies;
    ret->galaxy[galaxy_index]->infos->body_count = body_count;
    ret->galaxy[galaxy_index]->infos->map_limits.x = width;
    ret->galaxy[galaxy_index]->infos->map_limits.y = height;
    ret->galaxy[galaxy_index]->infos->side_cell_count_lowest_level = (unsigned long) sqrt(pow(4, max_depth));
    ret->galaxy[galaxy_index]->infos->max_local_work_size = CL_DEVICE_MAX_WORK_GROUP_SIZE;
    ret->galaxy[galaxy_index]->infos->last_layer_idx = cell_count - pow(4, max_depth);

    ret->galaxy[galaxy_index]->infos->cell_buffer_offset = ret->cell_buffer_offset[galaxy_index];
    ret->galaxy[galaxy_index]->infos->body_buffer_offset = ret->body_buffer_offset[galaxy_index];
    ret->galaxy[galaxy_index]->infos->history_buffer_offset = ret->history_buffer_offset[galaxy_index];

    ret->galaxy[galaxy_index]->infos->small_cell_size.x = (width /
                                                           ret->galaxy[galaxy_index]->infos->side_cell_count_lowest_level);
    ret->galaxy[galaxy_index]->infos->small_cell_size.y = (height /
                                                           ret->galaxy[galaxy_index]->infos->side_cell_count_lowest_level);

    if (cell_count > ret->highest_cell_count)
        ret->highest_cell_count = cell_count;

    if (body_count > ret->highest_body_count)
        ret->highest_body_count = body_count;

    if (max_depth > ret->highest_depth) {
        ret->highest_depth = max_depth;
        ret->highest_depth_last_layer_index = cell_count - pow(4, max_depth);
    }

    ret->cell_count[galaxy_index] = cell_count;
    ret->body_count[galaxy_index] = body_count;
    ret->max_local_work_size[galaxy_index] = CL_DEVICE_MAX_WORK_GROUP_SIZE;
    ret->depth[galaxy_index] = max_depth;
    ret->last_layer_idx[galaxy_index] = cell_count - pow(4, max_depth);

    unsigned int pattern = 0;

    clEnqueueWriteBuffer(ocl->queue, ret->cells, CL_TRUE, sizeof(cell) * ret->cell_buffer_offset[galaxy_index],
                         sizeof(cell) * cell_count, ret->galaxy[galaxy_index]->cells, 0, NULL,
                         NULL);
    clEnqueueWriteBuffer(ocl->queue, ret->bodies, CL_TRUE, sizeof(body) * ret->body_buffer_offset[galaxy_index],
                         sizeof(body) * body_count, ret->galaxy[galaxy_index]->bodies, 0, NULL,
                         NULL);
    clEnqueueWriteBuffer(ocl->queue, ret->infos, CL_TRUE, sizeof(galaxy_infos) * galaxy_index, sizeof(galaxy_infos),
                         ret->galaxy[galaxy_index]->infos, 0, NULL, NULL);
    clEnqueueFillBuffer(ocl->queue, ret->compute_history, &pattern, sizeof(unsigned int),
                        sizeof(unsigned int) * ret->history_buffer_offset[galaxy_index],
                        sizeof(unsigned int) * (body_count * cell_count), 0, NULL, NULL);
    clFinish(ocl->queue);

    *cells = ret->galaxy[galaxy_index]->cells;
    free(ret->galaxy[galaxy_index]->infos);
    free(ret->galaxy[galaxy_index]);
    ret->galaxy[galaxy_index] = NULL;
}

void galaxy_contains_lost(ocl_galaxy *galaxy, ocl *ocl, unsigned int *vals) {

    unsigned int val = 0;
    unsigned long dimensions[] = {
            galaxy->highest_body_count,
            galaxy->galaxy_count
    };

    for (size_t gidx = 0; gidx < galaxy->galaxy_count; ++gidx) {
        ocl->err = clEnqueueWriteBuffer(ocl->queue, galaxy->contains_losts, CL_TRUE, sizeof(unsigned int) * gidx,
                                        sizeof(unsigned int), &val, 0, NULL,
                                        NULL);
        if (ocl->err) {

            dprintf(STDERR_FILENO, "contains_losts %d: Unable to set contains_losts buffer to 0\n", ocl->err);
            exit(ocl->err);

        }
    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_CONTAINS_LOSTS], 0, sizeof(cl_mem), &galaxy->bodies);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "contains_losts %d: Unable to set argument 0 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_CONTAINS_LOSTS], 1, sizeof(cl_mem), &galaxy->infos);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "contains_losts %d: Unable to set argument 1 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_CONTAINS_LOSTS], 2, sizeof(cl_mem), &galaxy->contains_losts);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "contains_losts %d: Unable to set argument 2 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel[KERNEL_GALAXY_CONTAINS_LOSTS], 2, NULL,
                                      dimensions, NULL, 0, NULL, NULL);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "contains_losts %d: Error while calling kernel\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clEnqueueReadBuffer(ocl->queue, galaxy->contains_losts, CL_TRUE, 0,
                                   sizeof(unsigned int) * galaxy->galaxy_count, vals, 0,
                                   NULL,
                                   NULL);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "contains_losts %d: Unable to retrieve contains_losts value\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clFinish(ocl->queue);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "contains_losts %d: Error while finishing execution queue\n", ocl->err);
        exit(ocl->err);

    }

}

void galaxy_dispatch_all_losts(ocl_galaxy *galaxy, ocl *ocl) {

    unsigned long dimensions[] = {
            galaxy->highest_body_count,
            galaxy->galaxy_count
    };

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_DISPATCH_LOSTS], 0, sizeof(cl_mem), &galaxy->cells);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "dispatch_losts %d: Unable to set argument 0 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_DISPATCH_LOSTS], 1, sizeof(cl_mem), &galaxy->bodies);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "dispatch_losts %d: Unable to set argument 1 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_DISPATCH_LOSTS], 2, sizeof(cl_mem), &galaxy->infos);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "dispatch_losts %d: Unable to set argument 2 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel[KERNEL_GALAXY_DISPATCH_LOSTS], 2, NULL,
                                      dimensions, NULL, 0, NULL, NULL);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "dispatch_losts %d: Error while calling kernel\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clFinish(ocl->queue);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "dispatch_losts %d: Error while finishing execution queue\n", ocl->err);
        exit(ocl->err);

    }

}

void galaxy_contains_sub_dispatchables(ocl_galaxy *galaxy, ocl *ocl, unsigned int *vals) {

    unsigned int val = 0;
    unsigned long dimensions[] = {
            galaxy->highest_cell_count,
            galaxy->galaxy_count
    };

    for (size_t gidx = 0; gidx < galaxy->galaxy_count; ++gidx) {
        ocl->err = clEnqueueWriteBuffer(ocl->queue, galaxy->contains_sub_dispatchables, CL_TRUE,
                                        sizeof(unsigned int) * gidx,
                                        sizeof(unsigned int), &val, 0, NULL,
                                        NULL);
        if (ocl->err) {

            dprintf(STDERR_FILENO,
                    "contains_sub_dispatchables %d: Unable to set contains_sub_dispatchables buffer to 0\n", ocl->err);
            exit(ocl->err);

        }
    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_CONTAINS_SUB_DISPATCHABLES], 0, sizeof(cl_mem), &galaxy->cells);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "contains_sub_dispatchables %d: Unable to set argument 0 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_CONTAINS_SUB_DISPATCHABLES], 1, sizeof(cl_mem), &galaxy->infos);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "contains_sub_dispatchables %d: Unable to set argument 1 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_CONTAINS_SUB_DISPATCHABLES], 2, sizeof(cl_mem),
                              &galaxy->contains_sub_dispatchables);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "contains_sub_dispatchables %d: Unable to set argument 2 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel[KERNEL_GALAXY_CONTAINS_SUB_DISPATCHABLES], 2, NULL,
                                      dimensions, NULL, 0, NULL, NULL);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "contains_sub_dispatchables %d: Error while calling kernel\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clEnqueueReadBuffer(ocl->queue, galaxy->contains_sub_dispatchables, CL_TRUE, 0,
                                   sizeof(unsigned int) * galaxy->galaxy_count,
                                   vals, 0, NULL, NULL);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "contains_sub_dispatchables %d: Unable to retrieve contains_sub_dispatchables value\n",
                ocl->err);
        exit(ocl->err);

    }

    ocl->err = clFinish(ocl->queue);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "contains_sub_dispatchables %d: Error while finishing execution queue\n", ocl->err);
        exit(ocl->err);

    }

}

void galaxy_dispatch_all_sub_dispatchables(ocl_galaxy *galaxy, ocl *ocl) {

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_DISPATCH_SUB_DISPATCHABLES], 0, sizeof(cl_mem), &galaxy->cells);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "dispatch_sub_dispatchables %d: Unable to set argument 0 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_DISPATCH_SUB_DISPATCHABLES], 1, sizeof(cl_mem),
                              &galaxy->bodies);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "dispatch_sub_dispatchables %d: Unable to set argument 1 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_DISPATCH_SUB_DISPATCHABLES], 2, sizeof(cl_mem), &galaxy->infos);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "dispatch_sub_dispatchables %d: Unable to set argument 2 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    unsigned long start_idx = 0;
    unsigned long cell_count = 1;

    for (size_t idx = 0; idx < galaxy->highest_depth; ++idx) {

        const unsigned long dimensions[] = {
                cell_count,
                galaxy->galaxy_count
        };

        ocl->err = clEnqueueWriteBuffer(ocl->queue, galaxy->dispatch_sub_dispatchables_start_idx, CL_TRUE, 0,
                                        sizeof(unsigned int), &start_idx, 0, NULL, NULL);
        if (ocl->err) {

            dprintf(STDERR_FILENO,
                    "dispatch_sub_dispatchables %d: Unable to set dispatch_sub_dispatchables_start_idx buffer to 0\n",
                    ocl->err);
            exit(ocl->err);

        }

        ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_DISPATCH_SUB_DISPATCHABLES], 3, sizeof(cl_mem),
                                  &galaxy->dispatch_sub_dispatchables_start_idx);
        if (ocl->err) {

            dprintf(STDERR_FILENO, "dispatch_sub_dispatchables %d: Unable to set argument 3 for kernel call\n",
                    ocl->err);
            exit(ocl->err);

        }

        ocl->err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel[KERNEL_GALAXY_DISPATCH_SUB_DISPATCHABLES], 2, NULL,
                                          dimensions, NULL, 0, NULL, NULL);
        if (ocl->err) {

            dprintf(STDERR_FILENO, "dispatch_sub_dispatchables %d: Error while calling kernel\n", ocl->err);
            exit(ocl->err);

        }

        ocl->err = clFinish(ocl->queue);
        if (ocl->err) {

            dprintf(STDERR_FILENO, "dispatch_sub_dispatchables %d: Error while finishing execution queue\n", ocl->err);
            exit(ocl->err);

        }

        body_sort(galaxy, ocl);
        cell_clear_idxs(galaxy, ocl);
        cell_set_idxs(galaxy, ocl);
        cell_set_amount(galaxy, ocl);

        cell_count *= 4;
        start_idx += pow(4, idx);

    }

}

void galaxy_recover_bodies(ocl_galaxy *galaxy, ocl *ocl, body *bodies, unsigned long galaxy_index) {

    ocl->err = clEnqueueReadBuffer(ocl->queue, galaxy->bodies, CL_TRUE,
                                   sizeof(body) * galaxy->body_buffer_offset[galaxy_index],
                                   sizeof(body) * galaxy->body_count[galaxy_index], bodies, 0,
                                   NULL, NULL);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "recover_bodies %d: Unable to recover bodies from device memory\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clFinish(ocl->queue);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "recover_bodies %d: Error while finishing execution queue\n", ocl->err);
        exit(ocl->err);

    }

}

void galaxy_recover_cells(ocl_galaxy *galaxy, ocl *ocl, cell *cells, unsigned long galaxy_index) {

    ocl->err = clEnqueueReadBuffer(ocl->queue, galaxy->cells, CL_TRUE,
                                   sizeof(cell) * galaxy->cell_buffer_offset[galaxy_index],
                                   sizeof(cell) * galaxy->cell_count[galaxy_index], cells, 0,
                                   NULL, NULL);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "recover_cells %d: Unable to recover cells from device memory\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clFinish(ocl->queue);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "recover_cells %d: Error while finishing execution queue\n", ocl->err);
        exit(ocl->err);

    }

}

void galaxy_clear_inactive_cells(ocl_galaxy *galaxy, ocl *ocl) {

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_CLEAR_INACTIVE_CELLS], 0, sizeof(cl_mem), &galaxy->cells);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "clear_inactive_cells %d: Unable to set argument 0 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_CLEAR_INACTIVE_CELLS], 1, sizeof(cl_mem), &galaxy->infos);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "clear_inactive_cells %d: Unable to set argument 1 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    unsigned int start_idx = (unsigned int) galaxy->highest_depth_last_layer_index;
    unsigned long cell_count = (unsigned long) pow(4, galaxy->highest_depth);

    for (size_t idx = galaxy->highest_depth; idx > 0; --idx) {

        const unsigned long dimensions[] = {
            cell_count,
            galaxy->galaxy_count
        };

        ocl->err = clEnqueueWriteBuffer(ocl->queue, galaxy->clear_inactive_cells_start_idx, CL_TRUE, 0,
                                        sizeof(unsigned int), &start_idx, 0, NULL, NULL);
        if (ocl->err) {

            dprintf(STDERR_FILENO,
                    "clear_inactive_cells %d: Unable to set clear_inactive_cells_start_idx buffer to 0\n", ocl->err);
            exit(ocl->err);

        }

        ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_CLEAR_INACTIVE_CELLS], 2, sizeof(cl_mem),
                                  &galaxy->clear_inactive_cells_start_idx);
        if (ocl->err) {

            dprintf(STDERR_FILENO, "clear_inactive_cells %d: Unable to set argument 2 for kernel call\n", ocl->err);
            exit(ocl->err);

        }

        ocl->err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel[KERNEL_GALAXY_CLEAR_INACTIVE_CELLS], 2, NULL,
                                          dimensions, NULL, 0, NULL, NULL);
        if (ocl->err) {

            dprintf(STDERR_FILENO, "clear_inactive_cells %d: Error while calling kernel\n", ocl->err);
            exit(ocl->err);

        }

        ocl->err = clFinish(ocl->queue);
        if (ocl->err) {

            dprintf(STDERR_FILENO, "clear_inactive_cells %d: Error while finishing execution queue\n", ocl->err);
            exit(ocl->err);

        }

        cell_count /= 4;
        start_idx -= (unsigned long) pow(4, idx - 1);

    }

}

inline bool galaxy_vals_checks(unsigned int *vals, unsigned long galaxy_count) {
    for (size_t idx = 0; idx < galaxy_count; ++idx) {
        if (vals[idx])
            return true;
    }
    return false;
}


void galaxy_resolve(ocl_galaxy *galaxy, ocl *ocl) {

    unsigned int vals[galaxy->galaxy_count];

    galaxy_contains_lost(galaxy, ocl, vals);
    while (galaxy_vals_checks(vals, galaxy->galaxy_count)) {

        galaxy_dispatch_all_losts(galaxy, ocl);

        galaxy_contains_lost(galaxy, ocl, vals);
    }

    body_sort(galaxy, ocl);
    cell_clear_idxs(galaxy, ocl);
    cell_set_idxs(galaxy, ocl);
    cell_set_amount(galaxy, ocl);


    galaxy_contains_sub_dispatchables(galaxy, ocl, vals);
    while (galaxy_vals_checks(vals, galaxy->galaxy_count)) {

        galaxy_dispatch_all_sub_dispatchables(galaxy, ocl);

        galaxy_contains_sub_dispatchables(galaxy, ocl, vals);

    }

    galaxy_clear_inactive_cells(galaxy, ocl);

}

void galaxy_compute_com(ocl_galaxy *galaxy, ocl *ocl) {

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_COMPUTE_COM], 0, sizeof(cl_mem), &galaxy->cells);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "compute_com %d: Unable to set argument 0 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_COMPUTE_COM], 1, sizeof(cl_mem), &galaxy->bodies);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "compute_com %d: Unable to set argument 1 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_COMPUTE_COM], 2, sizeof(cl_mem), &galaxy->infos);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "compute_com %d: Unable to set argument 2 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    unsigned int start_idx = (unsigned int) galaxy->highest_depth_last_layer_index;
    unsigned long cell_count = (unsigned long) pow(4, galaxy->highest_depth);

    for (ssize_t idx = galaxy->highest_depth; idx >= 0; --idx) {

        const unsigned long dimensions[] = {
                cell_count,
                galaxy->galaxy_count
        };

        ocl->err = clEnqueueWriteBuffer(ocl->queue, galaxy->compute_com_start_idx, CL_TRUE, 0, sizeof(unsigned int),
                                        &start_idx, 0, NULL, NULL);
        if (ocl->err) {

            dprintf(STDERR_FILENO,
                    "compute_com %d: Unable to set compute_com_start_idx buffer to 0\n", ocl->err);
            exit(ocl->err);

        }

        ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_COMPUTE_COM], 3, sizeof(cl_mem),
                                  &galaxy->compute_com_start_idx);
        if (ocl->err) {

            dprintf(STDERR_FILENO, "compute_com %d: Unable to set argument 3 for kernel call\n", ocl->err);
            exit(ocl->err);

        }

        ocl->err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel[KERNEL_GALAXY_COMPUTE_COM], 2, NULL, dimensions,
                                          NULL, 0, NULL, NULL);
        if (ocl->err) {

            dprintf(STDERR_FILENO, "compute_com %d: Error while calling kernel\n", ocl->err);
            exit(ocl->err);

        }

        ocl->err = clFinish(ocl->queue);
        if (ocl->err) {

            dprintf(STDERR_FILENO, "compute_com %d: Error while finishing execution queue\n", ocl->err);
            exit(ocl->err);

        }

        cell_count /= 4;
        start_idx -= (unsigned long) pow(4, idx - 1);

    }

}

void galaxy_compute_accelerations(ocl_galaxy *galaxy, ocl *ocl) {

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_COMPUTE_ACCELERATIONS], 0, sizeof(cl_mem), &galaxy->cells);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "compute_accelerations %d: Unable to set argument 0 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_COMPUTE_ACCELERATIONS], 1, sizeof(cl_mem), &galaxy->bodies);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "compute_accelerations %d: Unable to set argument 1 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_COMPUTE_ACCELERATIONS], 2, sizeof(cl_mem), &galaxy->infos);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "compute_accelerations %d: Unable to set argument 2 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_COMPUTE_ACCELERATIONS], 3, sizeof(cl_mem),
                              &galaxy->compute_history);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "compute_accelerations %d: Unable to set argument 3 for kernel call\n", ocl->err);
        exit(ocl->err);

    }

    unsigned long start_idx = 0;
    unsigned long dimensions[] = {
            galaxy->highest_body_count,
            1,
            galaxy->galaxy_count
    };

    for (size_t idx = 0; idx < galaxy->highest_depth; ++idx) {

        ocl->err = clEnqueueWriteBuffer(ocl->queue, galaxy->compute_accelerations_start_idx, CL_TRUE, 0,
                                        sizeof(unsigned int), &start_idx, 0, NULL, NULL);
        if (ocl->err) {

            dprintf(STDERR_FILENO,
                    "compute_accelerations %d: Unable to set compute_accelerations_start_idx buffer to 0\n", ocl->err);
            exit(ocl->err);

        }

        ocl->err = clSetKernelArg(ocl->kernel[KERNEL_GALAXY_COMPUTE_ACCELERATIONS], 4, sizeof(cl_mem),
                                  &galaxy->compute_accelerations_start_idx);
        if (ocl->err) {

            dprintf(STDERR_FILENO, "compute_accelerations %d: Unable to set argument 4 for kernel call\n", ocl->err);
            exit(ocl->err);

        }

        ocl->err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel[KERNEL_GALAXY_COMPUTE_ACCELERATIONS], 3, NULL,
                                          dimensions, NULL, 0, NULL, NULL);
        if (ocl->err) {

            dprintf(STDERR_FILENO, "compute_accelerations %d: Error while calling kernel\n", ocl->err);
            exit(ocl->err);

        }

        ocl->err = clFinish(ocl->queue);
        if (ocl->err) {

            dprintf(STDERR_FILENO, "compute_accelerations %d: Error while finishing execution queue\n", ocl->err);
            exit(ocl->err);

        }

        dimensions[1] *= 4;
        start_idx += pow(4, idx);

    }

}

void galaxy_compute(ocl_galaxy *galaxy, ocl *ocl) {

    galaxy_compute_com(galaxy, ocl);
    galaxy_compute_accelerations(galaxy, ocl);
    body_apply_accelerations(galaxy, ocl);

    unsigned int pattern = 0;
    ocl->err = clEnqueueFillBuffer(ocl->queue, galaxy->compute_history, &pattern, sizeof(unsigned int), 0,
                                   sizeof(unsigned int) * galaxy->history_size, 0, NULL, NULL);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "galaxy_compute %d: Error while erasing compute_history\n", ocl->err);
        exit(ocl->err);

    }

    ocl->err = clFinish(ocl->queue);
    if (ocl->err) {

        dprintf(STDERR_FILENO, "galaxy_compute %d: Error while finishing execution queue\n", ocl->err);
        exit(ocl->err);

    }

}
